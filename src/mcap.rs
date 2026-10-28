//! An `.mcap` read through its summary and appended to in place.
//!
//! A recording is tens of gigabytes, so nothing here copies it. Reading goes through
//! the summary's chunk indexes, decompressing only the chunks that carry the channels
//! asked for. Appending puts new chunks where the data section ends (over the old
//! DataEnd and summary) and then writes a fresh summary after them, the same layout
//! `mcap_edit.py`'s `copy_topic_in` / `_write_tail` produce.
//!
//! Until a new summary is down, the last one written is the only way into the file, so
//! it is kept in memory: an error, a panic or a SIGINT/SIGTERM/SIGHUP part way through
//! puts it back (`restore`) and the file reads exactly as it did at the last commit.

use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Once};

pub const MAGIC: &[u8; 8] = b"\x89MCAP0\r\n";
const OP_SCHEMA: u8 = 0x03;
const OP_CHANNEL: u8 = 0x04;
const OP_MESSAGE: u8 = 0x05;
const OP_CHUNK: u8 = 0x06;
const OP_MESSAGE_INDEX: u8 = 0x07;
const OP_CHUNK_INDEX: u8 = 0x08;
const OP_ATTACHMENT_INDEX: u8 = 0x0A;
const OP_STATISTICS: u8 = 0x0B;
const OP_METADATA_INDEX: u8 = 0x0D;
const OP_SUMMARY_OFFSET: u8 = 0x0E;
const OP_FOOTER: u8 = 0x02;
const OP_DATA_END: u8 = 0x0F;
/// Private-range opcode readers skip; blanks the chunks of a removed output.
const OP_FILLER: u8 = 0x80;
const RECORD_OVERHEAD: usize = 9;
const FOOTER_LENGTH: u64 = 29;
const DATA_END_LENGTH: u64 = RECORD_OVERHEAD as u64 + 4;

// ---- records --------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Cursor<'a> {
        Cursor { data, at: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(count).filter(|end| *end <= self.data.len());
        let end = end.ok_or_else(|| "mcap record is shorter than its fields".to_string())?;
        let bytes = &self.data[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn text(&mut self) -> Result<String, String> {
        let length = self.u32()? as usize;
        String::from_utf8(self.take(length)?.to_vec()).map_err(|e| format!("mcap string: {e}"))
    }

    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let length = self.u32()? as usize;
        Ok(self.take(length)?.to_vec())
    }
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

pub fn record(opcode: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(RECORD_OVERHEAD + body.len());
    out.push(opcode);
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// Every `(opcode, body)` in a run of records (a decompressed chunk, a summary).
pub fn records(blob: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at + RECORD_OVERHEAD > blob.len() {
            return None;
        }
        let opcode = blob[at];
        let length = u64::from_le_bytes(blob[at + 1..at + 9].try_into().unwrap()) as usize;
        let start = at + RECORD_OVERHEAD;
        let end = start.saturating_add(length).min(blob.len());
        at = end;
        Some((opcode, &blob[start..end]))
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
    pub id: u16,
    pub name: String,
    pub encoding: String,
    pub data: Vec<u8>,
}

impl Schema {
    fn parse(body: &[u8]) -> Result<Schema, String> {
        let mut c = Cursor::new(body);
        Ok(Schema { id: c.u16()?, name: c.text()?, encoding: c.text()?, data: c.bytes()? })
    }

    fn encode(&self) -> Vec<u8> {
        let mut body = self.id.to_le_bytes().to_vec();
        put_text(&mut body, &self.name);
        put_text(&mut body, &self.encoding);
        body.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        body.extend_from_slice(&self.data);
        record(OP_SCHEMA, &body)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub id: u16,
    pub schema_id: u16,
    pub topic: String,
    pub message_encoding: String,
    pub metadata: Vec<(String, String)>,
}

impl Channel {
    fn parse(body: &[u8]) -> Result<Channel, String> {
        let mut c = Cursor::new(body);
        let (id, schema_id) = (c.u16()?, c.u16()?);
        let (topic, message_encoding) = (c.text()?, c.text()?);
        let pairs_length = c.u32()? as usize;
        let stop = c.at + pairs_length;
        let mut metadata = Vec::new();
        while c.at < stop {
            metadata.push((c.text()?, c.text()?));
        }
        Ok(Channel { id, schema_id, topic, message_encoding, metadata })
    }

    fn encode(&self) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&self.id.to_le_bytes());
        body.extend_from_slice(&self.schema_id.to_le_bytes());
        put_text(&mut body, &self.topic);
        put_text(&mut body, &self.message_encoding);
        let mut pairs = Vec::new();
        for (key, value) in &self.metadata {
            put_text(&mut pairs, key);
            put_text(&mut pairs, value);
        }
        body.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
        body.extend_from_slice(&pairs);
        record(OP_CHANNEL, &body)
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkIndex {
    pub message_start_time: u64,
    pub message_end_time: u64,
    pub chunk_start_offset: u64,
    pub chunk_length: u64,
    pub message_index_offsets: Vec<(u16, u64)>,
    pub message_index_length: u64,
    pub compression: String,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}

impl ChunkIndex {
    fn parse(body: &[u8]) -> Result<ChunkIndex, String> {
        let mut c = Cursor::new(body);
        let (message_start_time, message_end_time) = (c.u64()?, c.u64()?);
        let (chunk_start_offset, chunk_length) = (c.u64()?, c.u64()?);
        let offsets_length = c.u32()? as usize;
        let stop = c.at + offsets_length;
        let mut message_index_offsets = Vec::new();
        while c.at < stop {
            message_index_offsets.push((c.u16()?, c.u64()?));
        }
        Ok(ChunkIndex {
            message_start_time,
            message_end_time,
            chunk_start_offset,
            chunk_length,
            message_index_offsets,
            message_index_length: c.u64()?,
            compression: c.text()?,
            compressed_size: c.u64()?,
            uncompressed_size: c.u64()?,
        })
    }

    fn encode(&self) -> Vec<u8> {
        let mut body = Vec::new();
        for value in [self.message_start_time, self.message_end_time, self.chunk_start_offset, self.chunk_length] {
            body.extend_from_slice(&value.to_le_bytes());
        }
        body.extend_from_slice(&(self.message_index_offsets.len() as u32 * 10).to_le_bytes());
        for (channel, offset) in &self.message_index_offsets {
            body.extend_from_slice(&channel.to_le_bytes());
            body.extend_from_slice(&offset.to_le_bytes());
        }
        body.extend_from_slice(&self.message_index_length.to_le_bytes());
        put_text(&mut body, &self.compression);
        body.extend_from_slice(&self.compressed_size.to_le_bytes());
        body.extend_from_slice(&self.uncompressed_size.to_le_bytes());
        record(OP_CHUNK_INDEX, &body)
    }

    /// Whether this chunk may hold one of `channels` (a chunk indexed without message
    /// indexes says nothing, so it may hold anything).
    fn may_hold(&self, channels: &HashSet<u16>) -> bool {
        self.message_index_offsets.is_empty()
            || self.message_index_offsets.iter().any(|(id, _)| channels.contains(id))
    }

    fn span_end(&self) -> u64 {
        self.chunk_start_offset + self.chunk_length + self.message_index_length
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Statistics {
    pub message_count: u64,
    pub schema_count: u16,
    pub channel_count: u32,
    pub attachment_count: u32,
    pub metadata_count: u32,
    pub chunk_count: u32,
    pub message_start_time: u64,
    pub message_end_time: u64,
    pub channel_message_counts: BTreeMap<u16, u64>,
}

impl Statistics {
    fn parse(body: &[u8]) -> Result<Statistics, String> {
        let mut c = Cursor::new(body);
        let mut stats = Statistics {
            message_count: c.u64()?,
            schema_count: c.u16()?,
            channel_count: c.u32()?,
            attachment_count: c.u32()?,
            metadata_count: c.u32()?,
            chunk_count: c.u32()?,
            message_start_time: c.u64()?,
            message_end_time: c.u64()?,
            ..Default::default()
        };
        let counts_length = c.u32()? as usize;
        let stop = c.at + counts_length;
        while c.at < stop {
            let channel = c.u16()?;
            stats.channel_message_counts.insert(channel, c.u64()?);
        }
        Ok(stats)
    }

    fn encode(&self) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&self.message_count.to_le_bytes());
        body.extend_from_slice(&self.schema_count.to_le_bytes());
        for value in [self.channel_count, self.attachment_count, self.metadata_count, self.chunk_count] {
            body.extend_from_slice(&value.to_le_bytes());
        }
        body.extend_from_slice(&self.message_start_time.to_le_bytes());
        body.extend_from_slice(&self.message_end_time.to_le_bytes());
        body.extend_from_slice(&(self.channel_message_counts.len() as u32 * 10).to_le_bytes());
        for (channel, count) in &self.channel_message_counts {
            body.extend_from_slice(&channel.to_le_bytes());
            body.extend_from_slice(&count.to_le_bytes());
        }
        record(OP_STATISTICS, &body)
    }
}

/// What the summary section says about the file.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub schemas: BTreeMap<u16, Schema>,
    pub channels: BTreeMap<u16, Channel>,
    pub chunk_indexes: Vec<ChunkIndex>,
    pub statistics: Option<Statistics>,
    /// Attachment and metadata indexes, carried through verbatim.
    other: Vec<Vec<u8>>,
}

impl Summary {
    fn parse(blob: &[u8]) -> Result<Summary, String> {
        let mut summary = Summary::default();
        for (opcode, body) in records(blob) {
            match opcode {
                OP_SCHEMA => {
                    let schema = Schema::parse(body)?;
                    summary.schemas.insert(schema.id, schema);
                }
                OP_CHANNEL => {
                    let channel = Channel::parse(body)?;
                    summary.channels.insert(channel.id, channel);
                }
                OP_CHUNK_INDEX => summary.chunk_indexes.push(ChunkIndex::parse(body)?),
                OP_STATISTICS => summary.statistics = Some(Statistics::parse(body)?),
                OP_ATTACHMENT_INDEX | OP_METADATA_INDEX => summary.other.push(record(opcode, body)),
                _ => {}
            }
        }
        summary.chunk_indexes.sort_by_key(|index| index.chunk_start_offset);
        Ok(summary)
    }

    /// DataEnd, the summary, its offsets, the footer and the closing magic, for a data
    /// section that ends at `data_end`.
    fn encode_tail(&self, data_end: u64) -> Vec<u8> {
        let mut out = record(OP_DATA_END, &0u32.to_le_bytes());
        let summary_start = data_end + out.len() as u64;
        let mut groups: Vec<(u8, u64, u64)> = Vec::new();
        let mut group = |out: &mut Vec<u8>, opcode: u8, bytes: Vec<u8>| {
            if !bytes.is_empty() {
                groups.push((opcode, data_end + out.len() as u64, bytes.len() as u64));
                out.extend_from_slice(&bytes);
            }
        };
        let used: HashSet<u16> = self.channels.values().map(|channel| channel.schema_id).collect();
        let schemas: Vec<u8> = self.schemas.values().filter(|s| used.contains(&s.id)).flat_map(Schema::encode).collect();
        group(&mut out, OP_SCHEMA, schemas);
        group(&mut out, OP_CHANNEL, self.channels.values().flat_map(Channel::encode).collect());
        group(&mut out, OP_CHUNK_INDEX, self.chunk_indexes.iter().flat_map(ChunkIndex::encode).collect());
        for raw in &self.other {
            out.extend_from_slice(raw);
        }
        if let Some(stats) = &self.statistics {
            group(&mut out, OP_STATISTICS, stats.encode());
        }
        let summary_offset_start = data_end + out.len() as u64;
        for (opcode, start, length) in groups {
            let mut body = vec![opcode];
            body.extend_from_slice(&start.to_le_bytes());
            body.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&record(OP_SUMMARY_OFFSET, &body));
        }
        let mut footer = summary_start.to_le_bytes().to_vec();
        footer.extend_from_slice(&summary_offset_start.to_le_bytes());
        footer.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&record(OP_FOOTER, &footer));
        out.extend_from_slice(MAGIC);
        out
    }
}

pub fn decompress(compression: &str, data: &[u8], uncompressed_size: u64) -> Result<Vec<u8>, String> {
    match compression {
        "" => Ok(data.to_vec()),
        "zstd" => zstd::bulk::decompress(data, uncompressed_size as usize).map_err(|e| format!("zstd chunk: {e}")),
        "lz4" => {
            let mut out = Vec::with_capacity(uncompressed_size as usize);
            lz4_flex::frame::FrameDecoder::new(data).read_to_end(&mut out).map_err(|e| format!("lz4 chunk: {e}"))?;
            Ok(out)
        }
        other => Err(format!("unsupported chunk compression {other:?}")),
    }
}

fn compress(compression: &str, data: &[u8]) -> Result<Vec<u8>, String> {
    match compression {
        "" => Ok(data.to_vec()),
        "zstd" => zstd::bulk::compress(data, 3).map_err(|e| format!("zstd: {e}")),
        "lz4" => {
            let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
            encoder.write_all(data).map_err(|e| e.to_string())?;
            encoder.finish().map_err(|e| e.to_string())
        }
        other => Err(format!("unsupported chunk compression {other:?}")),
    }
}

/// One message as read: `data` is the payload past the message record's header.
pub struct Message<'a> {
    pub channel_id: u16,
    pub log_time: u64,
    pub data: &'a [u8],
}

// ---- the file -------------------------------------------------------------------------

/// A valid ending for the file and where it goes: what `restore` puts back.
struct Tail {
    at: u64,
    bytes: Vec<u8>,
}

/// The part of the file the signal handler needs. Every write takes this lock, so a
/// restore never interleaves with one.
struct Guard {
    file: Option<File>,
    committed: Tail,
    /// Bytes past `committed.at` have been written since the last commit.
    dirty: bool,
}

impl Guard {
    fn restore(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        let file = self.file.as_mut().ok_or("no file to restore")?;
        file.seek(SeekFrom::Start(self.committed.at)).map_err(|e| e.to_string())?;
        file.write_all(&self.committed.bytes).map_err(|e| e.to_string())?;
        file.set_len(self.committed.at + self.committed.bytes.len() as u64).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        self.dirty = false;
        Ok(())
    }
}

static GUARDS: Mutex<Vec<(PathBuf, Arc<Mutex<Guard>>)>> = Mutex::new(Vec::new());
static SIGNALS: Once = Once::new();

/// On SIGINT/SIGTERM/SIGHUP, put back the last summary of every file mid-write, then exit.
fn install_signal_restore() {
    SIGNALS.call_once(|| {
        use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
        let Ok(mut signals) = signal_hook::iterator::Signals::new([SIGINT, SIGTERM, SIGHUP]) else {
            eprintln!("WARNING: could not install the signal handler that restores an interrupted .mcap");
            return;
        };
        std::thread::spawn(move || {
            if let Some(signal) = signals.forever().next() {
                let guards = GUARDS.lock().map(|g| g.clone()).unwrap_or_default();
                // every lock stays held until exit, so no write can start after its restore
                let mut held = Vec::new();
                for (path, guard) in &guards {
                    let mut guard = guard.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    if guard.dirty {
                        match guard.restore() {
                            Ok(()) => eprintln!("interrupted: {} is back to its last summary", path.display()),
                            Err(error) => eprintln!("interrupted, and could not restore {}: {error}", path.display()),
                        }
                    }
                    held.push(guard);
                }
                std::process::exit(128 + signal);
            }
        });
    });
}

pub struct McapFile {
    path: PathBuf,
    reader: File,
    /// Where the data section ends: the next chunk goes here, the DataEnd after it.
    data_end: u64,
    pub summary: Summary,
    committed_summary: Summary,
    committed_data_end: u64,
    guard: Arc<Mutex<Guard>>,
}

impl McapFile {
    pub fn open(path: &Path) -> Result<McapFile, String> {
        let mut reader = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let size = reader.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        let read_at = |reader: &mut File, offset: u64, length: usize| -> Result<Vec<u8>, String> {
            let mut buffer = vec![0u8; length];
            reader.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
            reader.read_exact(&mut buffer).map_err(|e| e.to_string())?;
            Ok(buffer)
        };
        let shown = path.display();
        if size < 8 + FOOTER_LENGTH + 8 || read_at(&mut reader, 0, 8)? != MAGIC {
            return Err(format!("{shown}: not an mcap"));
        }
        if read_at(&mut reader, size - 8, 8)? != MAGIC {
            return Err(format!("{shown}: no closing magic (an unfinished recording?) -- run mcap_recover first"));
        }
        let footer = read_at(&mut reader, size - 8 - FOOTER_LENGTH, FOOTER_LENGTH as usize)?;
        if footer[0] != OP_FOOTER {
            return Err(format!("{shown}: the footer is not where the spec puts it"));
        }
        let summary_start = u64::from_le_bytes(footer[9..17].try_into().unwrap());
        let summary_offset_start = u64::from_le_bytes(footer[17..25].try_into().unwrap());
        if summary_start == 0 {
            return Err(format!("{shown}: has no summary section to read it by"));
        }
        let data_end = summary_start - DATA_END_LENGTH;
        if read_at(&mut reader, data_end, 1)?[0] != OP_DATA_END {
            return Err(format!("{shown}: no DataEnd record before the summary"));
        }
        let summary_end = if summary_offset_start != 0 { summary_offset_start } else { size - 8 - FOOTER_LENGTH };
        let summary = Summary::parse(&read_at(&mut reader, summary_start, (summary_end - summary_start) as usize)?)?;
        if summary.chunk_indexes.is_empty() && summary.statistics.as_ref().is_some_and(|s| s.message_count > 0) {
            return Err(format!("{shown}: its messages are not in indexed chunks, which this reader needs"));
        }
        let tail = read_at(&mut reader, data_end, (size - data_end) as usize)?;
        let guard = Guard { file: None, committed: Tail { at: data_end, bytes: tail }, dirty: false };
        Ok(McapFile {
            path: path.to_path_buf(),
            reader,
            data_end,
            committed_summary: summary.clone(),
            summary,
            committed_data_end: data_end,
            guard: Arc::new(Mutex::new(guard)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A chunk's records, decompressed.
    pub fn read_chunk(&mut self, index: &ChunkIndex) -> Result<Vec<u8>, String> {
        let mut raw = vec![0u8; index.chunk_length as usize];
        self.reader.seek(SeekFrom::Start(index.chunk_start_offset)).map_err(|e| e.to_string())?;
        self.reader.read_exact(&mut raw).map_err(|e| format!("reading a chunk: {e}"))?;
        if raw.first() != Some(&OP_CHUNK) {
            return Err(format!("chunk index points at a {:#x} record, not a chunk", raw.first().unwrap_or(&0)));
        }
        let mut c = Cursor::new(&raw[RECORD_OVERHEAD..]);
        let (_, _) = (c.u64()?, c.u64()?);
        let uncompressed_size = c.u64()?;
        let _crc = c.u32()?;
        let compression = c.text()?;
        let records_length = c.u64()? as usize;
        decompress(&compression, c.take(records_length)?, uncompressed_size)
    }

    /// Every message on `channels` with a log time in `[start, end]`, in log-time order
    /// (chunks may overlap in time, so they are merged rather than concatenated).
    pub fn for_each_message(
        &mut self,
        channels: &HashSet<u16>,
        start: u64,
        end: u64,
        mut visit: impl FnMut(Message) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut chunks: Vec<ChunkIndex> = self
            .summary
            .chunk_indexes
            .iter()
            .filter(|index| index.may_hold(channels) && index.message_end_time >= start && index.message_start_time <= end)
            .cloned()
            .collect();
        chunks.sort_by_key(|index| (index.message_start_time, index.chunk_start_offset));
        // min-heap on (log time, arrival order)
        let mut pending: BinaryHeap<std::cmp::Reverse<(u64, u64, u16, Vec<u8>)>> = BinaryHeap::new();
        let mut arrival = 0u64;
        // emits everything earlier than `limit`, or everything when it is None
        let mut emit_before = |pending: &mut BinaryHeap<std::cmp::Reverse<(u64, u64, u16, Vec<u8>)>>, limit: Option<u64>| {
            while pending.peek().is_some_and(|std::cmp::Reverse(top)| limit.is_none_or(|limit| top.0 < limit)) {
                let std::cmp::Reverse((log_time, _, channel_id, data)) = pending.pop().unwrap();
                visit(Message { channel_id, log_time, data: &data })?;
            }
            Ok::<(), String>(())
        };
        for index in &chunks {
            emit_before(&mut pending, Some(index.message_start_time))?;
            let blob = self.read_chunk(index)?;
            for (opcode, body) in records(&blob) {
                if opcode != OP_MESSAGE || body.len() < 22 {
                    continue;
                }
                let channel_id = u16::from_le_bytes([body[0], body[1]]);
                let log_time = u64::from_le_bytes(body[6..14].try_into().unwrap());
                if channels.contains(&channel_id) && log_time >= start && log_time <= end {
                    pending.push(std::cmp::Reverse((log_time, arrival, channel_id, body[22..].to_vec())));
                    arrival += 1;
                }
            }
        }
        emit_before(&mut pending, None)
    }

    // ---- writing ----

    fn with_writer<T>(&mut self, act: impl FnOnce(&mut File) -> Result<T, String>) -> Result<T, String> {
        let mut guard = self.guard.lock().map_err(|_| "mcap write lock poisoned".to_string())?;
        if guard.file.is_none() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.path)
                .map_err(|e| format!("cannot open {} for writing: {e}", self.path.display()))?;
            guard.file = Some(file);
            GUARDS.lock().map_err(|_| "guard list poisoned")?.push((self.path.clone(), self.guard.clone()));
            install_signal_restore();
        }
        guard.dirty = true;
        act(guard.file.as_mut().unwrap())
    }

    pub fn compression(&self) -> String {
        self.summary.chunk_indexes.first().map(|index| index.compression.clone()).unwrap_or_else(|| "zstd".into())
    }

    /// Write one chunk of `records` (all on `channel_id`) and its message index at the end
    /// of the data section. `entries` are `(log_time, offset into records)` per message.
    pub fn append_chunk(&mut self, channel_id: u16, records_blob: &[u8], entries: &[(u64, u64)]) -> Result<(), String> {
        let Some(start) = entries.iter().map(|e| e.0).min() else {
            return Ok(());
        };
        let end = entries.iter().map(|e| e.0).max().unwrap();
        let compression = self.compression();
        let compressed = compress(&compression, records_blob)?;
        let mut body = Vec::with_capacity(compressed.len() + 64);
        body.extend_from_slice(&start.to_le_bytes());
        body.extend_from_slice(&end.to_le_bytes());
        body.extend_from_slice(&(records_blob.len() as u64).to_le_bytes());
        body.extend_from_slice(&crc32fast::hash(records_blob).to_le_bytes());
        put_text(&mut body, &compression);
        body.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        body.extend_from_slice(&compressed);
        let chunk = record(OP_CHUNK, &body);
        let mut index_body = channel_id.to_le_bytes().to_vec();
        index_body.extend_from_slice(&(entries.len() as u32 * 16).to_le_bytes());
        for (log_time, offset) in entries {
            index_body.extend_from_slice(&log_time.to_le_bytes());
            index_body.extend_from_slice(&offset.to_le_bytes());
        }
        let message_index = record(OP_MESSAGE_INDEX, &index_body);
        let at = self.data_end;
        self.with_writer(|file| {
            file.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            file.write_all(&message_index).map_err(|e| e.to_string())
        })?;
        self.summary.chunk_indexes.push(ChunkIndex {
            message_start_time: start,
            message_end_time: end,
            chunk_start_offset: at,
            chunk_length: chunk.len() as u64,
            message_index_offsets: vec![(channel_id, at + chunk.len() as u64)],
            message_index_length: message_index.len() as u64,
            compression,
            compressed_size: compressed.len() as u64,
            uncompressed_size: records_blob.len() as u64,
        });
        self.data_end = at + chunk.len() as u64 + message_index.len() as u64;
        if let Some(stats) = self.summary.statistics.as_mut() {
            stats.chunk_count += 1;
            *stats.channel_message_counts.entry(channel_id).or_insert(0) += entries.len() as u64;
            stats.message_count += entries.len() as u64;
            if stats.message_count == entries.len() as u64 {
                (stats.message_start_time, stats.message_end_time) = (start, end);
            } else {
                stats.message_start_time = stats.message_start_time.min(start);
                stats.message_end_time = stats.message_end_time.max(end);
            }
        }
        Ok(())
    }

    /// Declare a schema (or find the one already declared with this name and encoding).
    pub fn add_schema(&mut self, name: &str, encoding: &str, data: &[u8]) -> u16 {
        if let Some(found) = self.summary.schemas.values().find(|s| s.name == name && s.encoding == encoding) {
            return found.id;
        }
        let id = self.summary.schemas.keys().max().map_or(1, |max| max + 1);
        self.summary.schemas.insert(id, Schema { id, name: name.into(), encoding: encoding.into(), data: data.to_vec() });
        id
    }

    /// Declare a channel; returns its id and the channel (+ schema) records the first chunk
    /// on it should open with, so a reader walking the data section meets them first.
    pub fn add_channel(&mut self, mut channel: Channel) -> Result<(u16, Vec<u8>), String> {
        let id = match self.summary.channels.keys().max() {
            None => 0,
            Some(max) => max.checked_add(1).ok_or("this mcap has used every channel id")?,
        };
        channel.id = id;
        let mut prelude = Vec::new();
        if let Some(schema) = self.summary.schemas.get(&channel.schema_id) {
            prelude.extend_from_slice(&schema.encode());
        }
        prelude.extend_from_slice(&channel.encode());
        self.summary.channels.insert(id, channel);
        Ok((id, prelude))
    }

    fn refresh_statistics(&mut self) {
        let schema_count = {
            let used: HashSet<u16> = self.summary.channels.values().map(|c| c.schema_id).filter(|id| *id != 0).collect();
            used.len() as u16
        };
        let channel_count = self.summary.channels.len() as u32;
        if let Some(stats) = self.summary.statistics.as_mut() {
            stats.schema_count = schema_count;
            stats.channel_count = channel_count;
        }
    }

    /// Write the summary for everything appended so far; the file is complete and valid
    /// again, and this is now what an interruption puts back.
    pub fn commit(&mut self) -> Result<(), String> {
        let dirty = self.guard.lock().map_err(|_| "mcap write lock poisoned")?.dirty;
        if !dirty && self.data_end == self.committed_data_end && self.summary.channels == self.committed_summary.channels {
            return Ok(());
        }
        self.refresh_statistics();
        let tail = self.summary.encode_tail(self.data_end);
        let at = self.data_end;
        self.with_writer(|file| {
            file.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
            file.write_all(&tail).map_err(|e| e.to_string())?;
            file.set_len(at + tail.len() as u64).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())
        })?;
        let mut guard = self.guard.lock().map_err(|_| "mcap write lock poisoned")?;
        guard.committed = Tail { at, bytes: tail };
        guard.dirty = false;
        drop(guard);
        self.committed_summary = self.summary.clone();
        self.committed_data_end = self.data_end;
        Ok(())
    }

    /// Put the last committed summary back (after an error part way through a write).
    pub fn restore(&mut self) -> Result<(), String> {
        self.guard.lock().map_err(|_| "mcap write lock poisoned")?.restore()?;
        self.summary = self.committed_summary.clone();
        self.data_end = self.committed_data_end;
        Ok(())
    }

    /// Remove channels this tool appended (each alone in its own chunks): drop them from
    /// the summary and commit, then give back the space their chunks held -- truncating
    /// when they end the data section, blanking them with a skippable record otherwise.
    /// A channel that shares a chunk with anything else is part of the recording and is
    /// refused: removing it would mean rewriting the recording's chunks.
    pub fn remove_channels(&mut self, ids: &HashSet<u16>) -> Result<(), String> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for index in &self.summary.chunk_indexes {
            let holds: Vec<u16> = index.message_index_offsets.iter().map(|(id, _)| *id).collect();
            let mine = holds.iter().filter(|id| ids.contains(id)).count();
            if mine == 0 {
                if holds.is_empty() {
                    return Err("a chunk without message indexes might hold the topic being replaced; \
                                refusing to edit this file in place"
                        .into());
                }
                continue;
            }
            if mine != holds.len() {
                let topic = ids.iter().filter_map(|id| self.summary.channels.get(id)).map(|c| c.topic.clone()).next();
                return Err(format!(
                    "'{}' shares chunks with the recording's own topics, so it cannot be replaced in \
                     place; remove it with mcap_edit --delete first",
                    topic.unwrap_or_default()
                ));
            }
            spans.push((index.chunk_start_offset, index.span_end()));
        }
        let removed: u64 = ids.iter().map(|id| {
            self.summary.statistics.as_ref().and_then(|s| s.channel_message_counts.get(id)).copied().unwrap_or(0)
        }).sum();
        let chunk_count = spans.len() as u32;
        self.summary.chunk_indexes.retain(|index| !index.message_index_offsets.iter().any(|(id, _)| ids.contains(id)));
        self.summary.channels.retain(|id, _| !ids.contains(id));
        if let Some(stats) = self.summary.statistics.as_mut() {
            stats.channel_message_counts.retain(|id, _| !ids.contains(id));
            stats.message_count = stats.message_count.saturating_sub(removed);
            stats.chunk_count = stats.chunk_count.saturating_sub(chunk_count);
            if let (Some(first), Some(last)) = (
                self.summary.chunk_indexes.iter().map(|i| i.message_start_time).min(),
                self.summary.chunk_indexes.iter().map(|i| i.message_end_time).max(),
            ) {
                (stats.message_start_time, stats.message_end_time) = (first, last);
            }
        }
        // the summary stops pointing at them first; only then is it safe to touch their bytes
        self.with_writer(|_| Ok(()))?;
        self.commit()?;
        spans.sort();
        let mut reclaimed = false;
        while let Some(&(start, end)) = spans.last() {
            if end != self.data_end {
                break;
            }
            self.data_end = start;
            spans.pop();
            reclaimed = true;
        }
        if reclaimed {
            self.with_writer(|_| Ok(()))?;
            self.commit()?;
        }
        for (start, end) in spans {
            let length = end - start;
            let mut filler = vec![0u8; length as usize];
            filler[0] = OP_FILLER;
            filler[1..9].copy_from_slice(&(length - RECORD_OVERHEAD as u64).to_le_bytes());
            self.with_writer(|file| {
                file.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
                file.write_all(&filler).map_err(|e| e.to_string())
            })?;
        }
        let mut guard = self.guard.lock().map_err(|_| "mcap write lock poisoned")?;
        if let Some(file) = guard.file.as_mut() {
            file.sync_all().map_err(|e| e.to_string())?;
        }
        guard.dirty = false;
        Ok(())
    }
}

impl Drop for McapFile {
    fn drop(&mut self) {
        let Ok(mut guard) = self.guard.lock() else {
            return;
        };
        if guard.dirty {
            match guard.restore() {
                Ok(()) => eprintln!("{}: write did not finish; its last summary is restored", self.path.display()),
                Err(error) => eprintln!("{}: write did not finish, and restoring failed: {error}", self.path.display()),
            }
        }
        guard.file = None;
        if let Ok(mut guards) = GUARDS.lock() {
            guards.retain(|(_, other)| !Arc::ptr_eq(other, &self.guard));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid mcap: header, one zstd chunk holding a schema, channel and two
    /// messages, its message index, then the summary this module writes.
    pub(crate) fn tiny_mcap(path: &Path) {
        let mut out = MAGIC.to_vec();
        let mut header = Vec::new();
        put_text(&mut header, "ros2");
        put_text(&mut header, "test");
        out.extend_from_slice(&record(0x01, &header));
        let mut file = File::create(path).unwrap();
        file.write_all(&out).unwrap();
        drop(file);
        // an empty data section with an empty summary...
        let mut empty = File::options().append(true).open(path).unwrap();
        let summary = Summary { statistics: Some(Statistics::default()), ..Default::default() };
        empty.write_all(&summary.encode_tail(out.len() as u64)).unwrap();
        drop(empty);
        // ...and then a recording channel appended the same way outputs are
        let mut mcap = McapFile::open(path).unwrap();
        let schema = mcap.add_schema("std_msgs/msg/String", "ros2msg", b"string data\n");
        let (id, mut blob) = mcap
            .add_channel(Channel { id: 0, schema_id: schema, topic: "/chatter".into(), message_encoding: "cdr".into(), metadata: vec![] })
            .unwrap();
        let mut entries = Vec::new();
        for (time, text) in [(20u64, "b"), (10, "a")] {
            entries.push((time, blob.len() as u64));
            let mut body = id.to_le_bytes().to_vec();
            body.extend_from_slice(&0u32.to_le_bytes());
            body.extend_from_slice(&time.to_le_bytes());
            body.extend_from_slice(&time.to_le_bytes());
            body.extend_from_slice(text.as_bytes());
            blob.extend_from_slice(&record(OP_MESSAGE, &body));
        }
        mcap.append_chunk(id, &blob, &entries).unwrap();
        mcap.commit().unwrap();
    }

    fn texts(mcap: &mut McapFile, channel: u16) -> Vec<(u64, String)> {
        let mut out = Vec::new();
        mcap.for_each_message(&HashSet::from([channel]), 0, u64::MAX, |m| {
            out.push((m.log_time, String::from_utf8(m.data.to_vec()).unwrap()));
            Ok(())
        })
        .unwrap();
        out
    }

    #[test]
    fn appends_reads_back_in_time_order_and_removes() {
        let dir = std::env::temp_dir().join(format!("icp_stitch_mcap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.mcap");
        tiny_mcap(&path);
        let size_before = std::fs::metadata(&path).unwrap().len();
        let mut mcap = McapFile::open(&path).unwrap();
        assert_eq!(texts(&mut mcap, 0), vec![(10, "a".into()), (20, "b".into())]);
        assert_eq!(mcap.summary.statistics.as_ref().unwrap().channel_message_counts[&0], 2);

        // a second channel, then remove it: the file is back to the size it was
        let (id, mut blob) = mcap
            .add_channel(Channel { id: 0, schema_id: 0, topic: "/out".into(), message_encoding: "lcm".into(), metadata: vec![] })
            .unwrap();
        assert_eq!(id, 1);
        let mut body = id.to_le_bytes().to_vec();
        body.extend_from_slice(&[0; 20]);
        body.extend_from_slice(b"x");
        let offset = blob.len() as u64;
        blob.extend_from_slice(&record(OP_MESSAGE, &body));
        mcap.append_chunk(id, &blob, &[(0, offset)]).unwrap();
        mcap.commit().unwrap();
        drop(mcap);
        let mut mcap = McapFile::open(&path).unwrap();
        assert_eq!(texts(&mut mcap, 1), vec![(0, "x".into())]);
        assert!(std::fs::metadata(&path).unwrap().len() > size_before);
        mcap.remove_channels(&HashSet::from([1])).unwrap();
        drop(mcap);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), size_before);
        let mut mcap = McapFile::open(&path).unwrap();
        assert_eq!(mcap.summary.channels.len(), 1);
        assert_eq!(texts(&mut mcap, 0).len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unfinished_write_is_put_back() {
        let dir = std::env::temp_dir().join(format!("icp_stitch_mcap_restore_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.mcap");
        tiny_mcap(&path);
        let original = std::fs::read(&path).unwrap();
        {
            let mut mcap = McapFile::open(&path).unwrap();
            let (id, blob) = mcap
                .add_channel(Channel { id: 0, schema_id: 0, topic: "/half".into(), message_encoding: "lcm".into(), metadata: vec![] })
                .unwrap();
            mcap.append_chunk(id, &[blob, vec![0u8; 100_000]].concat(), &[(1, 0)]).unwrap();
            // no commit: dropping it (as an error unwinding out of a run does) restores
        }
        assert_eq!(std::fs::read(&path).unwrap(), original);
        McapFile::open(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
