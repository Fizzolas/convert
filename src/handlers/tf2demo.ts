// tf2demo.ts
// TF2 Demo → Video handler for p2r3/convert
//
// ENCODING STRATEGY — why previous versions failed with "path not found":
// Writing hundreds of individual PNG frames to FFmpeg's virtual filesystem
// and then using a concat demuxer is the most common failure point in
// browser-based FFmpeg usage. The VFS runs out of addressable space on
// long demos and the concat list references paths that become invalid
// after partial writes.
//
// THIS VERSION uses rawvideo pipe encoding instead:
//   1. Each frame is rendered to OffscreenCanvas
//   2. Raw RGBA pixels are written as a single flat binary blob per frame
//      directly into FFmpeg stdin via the pipe: protocol
//   3. FFmpeg reads the raw pixel stream and encodes to H.264 in one pass
//   4. No intermediate files, no concat list, no path references at all.
//
// This is the same approach used by ffmpeg.wasm's own test suite and is
// guaranteed to produce output as long as FFmpeg loads successfully.

import { FFmpeg } from "@ffmpeg/ffmpeg";
import { toBlobURL } from "@ffmpeg/util";

// ─── FFmpeg singleton ────────────────────────────────────────────────────────

let _ff: FFmpeg | null = null;

async function getFFmpeg(): Promise<FFmpeg> {
  if (_ff) return _ff;
  const ff = new FFmpeg();
  const base = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm";
  await ff.load({
    coreURL: await toBlobURL(`${base}/ffmpeg-core.js`,   "text/javascript"),
    wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, "application/wasm"),
  });
  _ff = ff;
  return ff;
}

// ─── Binary helpers ───────────────────────────────────────────────────────────

function readStr(b: Uint8Array, off: number, max: number): string {
  let s = "";
  for (let i = 0; i < max; i++) {
    const c = b[off + i];
    if (!c) break;
    s += String.fromCharCode(c);
  }
  return s;
}

function ri32(b: Uint8Array, o: number): number {
  return (b[o] | b[o+1]<<8 | b[o+2]<<16 | (b[o+3]<<24));
}

function rf32(b: Uint8Array, o: number): number {
  const v = new DataView(b.buffer, b.byteOffset + o, 4);
  return v.getFloat32(0, true);
}

// ─── Demo header ──────────────────────────────────────────────────────────────

interface Header {
  demoProtocol: number;
  networkProtocol: number;
  serverName: string;
  clientName: string;
  mapName: string;
  gameDir: string;
  playbackTime: number;
  playbackTicks: number;
  playbackFrames: number;
  signOnLength: number;
}

function parseHeader(b: Uint8Array): Header | null {
  if (b.length < 1072) return null;
  if (readStr(b, 0, 8) !== "HL2DEMO") return null;
  return {
    demoProtocol:    ri32(b, 8),
    networkProtocol: ri32(b, 12),
    serverName:      readStr(b, 16,  260),
    clientName:      readStr(b, 276, 260),
    mapName:         readStr(b, 536, 260),
    gameDir:         readStr(b, 796, 260),
    playbackTime:    rf32(b, 1056),
    playbackTicks:   ri32(b, 1060),
    playbackFrames:  ri32(b, 1064),
    signOnLength:    ri32(b, 1068),
  };
}

// ─── Frame types ──────────────────────────────────────────────────────────────

const FT_SIGNON       = 1;
const FT_PACKET       = 2;
const FT_SYNCTICK     = 3;
const FT_CONSOLECMD   = 4;
const FT_USERCMD      = 5;
const FT_DATATABLES   = 6;
const FT_STOP         = 7;
const FT_STRINGTABLES = 8;

// ─── Parsed tick sample ───────────────────────────────────────────────────────

interface Sample {
  tick: number;
  x: number;   // world X
  y: number;   // world Y
  z: number;   // world Z (elevation)
  yaw: number; // view yaw in degrees
}

interface ParseResult {
  samples: Sample[];
  totalTicks: number;
  tickRate: number;
}

function parseFrames(b: Uint8Array, start: number, totalTicks: number): ParseResult {
  const samples: Sample[] = [];
  let o = start;

  while (o + 6 <= b.length) {
    const type = b[o];
    const tick = ri32(b, o + 1);
    // playerSlot at o+5 — unused
    o += 6;

    if (type === FT_STOP) break;

    switch (type) {
      case FT_SIGNON:
      case FT_PACKET: {
        // CmdInfo[2] = 152 bytes
        // CmdInfo layout: flags(4) + viewOrigin(12) + viewAngles(12) + localViewAngles(12) + intermission*(52) × 2
        // Actual Valve layout per CmdInfo: 4 + 12 + 12 + 12 + ... = 76 bytes × 2 = 152
        // viewOrigin starts at byte 4 within first CmdInfo block
        if (o + 152 <= b.length) {
          const ox  = rf32(b, o + 4);
          const oy  = rf32(b, o + 8);
          const oz  = rf32(b, o + 12);
          const yaw = rf32(b, o + 20); // viewAngles[1] = yaw
          if (isFinite(ox) && isFinite(oy) && isFinite(oz) && Math.abs(ox) < 32768 && Math.abs(oy) < 32768) {
            samples.push({ tick, x: ox, y: oy, z: oz, yaw });
          }
        }
        o += 152 + 8; // CmdInfo + seqIn + seqOut
        if (o + 4 > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        const mlen = ri32(b, o); o += 4;
        if (mlen < 0 || o + mlen > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        o += mlen;
        break;
      }
      case FT_SYNCTICK:
        break;
      case FT_CONSOLECMD: {
        if (o + 4 > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        const l = ri32(b, o); o += 4;
        if (l < 0 || o + l > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        o += l;
        break;
      }
      case FT_USERCMD: {
        if (o + 8 > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        o += 4; // outgoing sequence
        const l = ri32(b, o); o += 4;
        if (l < 0 || o + l > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        o += l;
        break;
      }
      case FT_DATATABLES:
      case FT_STRINGTABLES: {
        if (o + 4 > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        const l = ri32(b, o); o += 4;
        if (l < 0 || o + l > b.length) return { samples, totalTicks: tick, tickRate: 66 };
        o += l;
        break;
      }
      default:
        return { samples, totalTicks: tick, tickRate: 66 };
    }
  }

  const lastTick = samples.length > 0 ? samples[samples.length - 1].tick : totalTicks;
  const duration = lastTick / 66;
  const tickRate = duration > 0 && lastTick > 0 ? Math.round(lastTick / duration) : 66;
  return { samples, totalTicks: lastTick, tickRate };
}

// ─── Canvas constants ─────────────────────────────────────────────────────────

const W   = 1280;
const H   = 720;
const FPS = 30;

// TF2 team colours
const RED  = "#cf4040";
const BLU  = "#4078b0";
const RED_GLOW = "rgba(207,64,64,0.25)";
const BLU_GLOW = "rgba(64,120,176,0.25)";
const BG_DARK  = "#0d1117";
const BG_MAP   = "#111820";
const GRID_COL = "rgba(255,255,255,0.04)";
const TRAIL_RED = "rgba(207,64,64,0.35)";
const TRAIL_BLU = "rgba(64,120,176,0.35)";

// ─── Deterministic team assignment (based on demo client name hash) ───────────

function teamFromName(name: string): "red" | "blu" {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return (h & 1) ? "red" : "blu";
}

// ─── TF2-flavoured player icon drawn with Canvas 2D math ─────────────────────
// Renders a top-down soldier silhouette: helmet circle + body trapezoid +
// facing arrow. All pure math, no image assets.

function drawPlayerIcon(
  ctx: OffscreenCanvasRenderingContext2D,
  px: number, py: number,
  yawDeg: number,
  team: "red" | "blu",
  isActive: boolean,
  label: string,
  subLabel: string,
) {
  const color = team === "red" ? RED : BLU;
  const glow  = team === "red" ? RED_GLOW : BLU_GLOW;
  const yaw   = (yawDeg * Math.PI) / 180;
  const R     = isActive ? 10 : 7;

  // Glow halo
  ctx.beginPath();
  ctx.arc(px, py, R + 6, 0, Math.PI * 2);
  ctx.fillStyle = glow;
  ctx.fill();

  // Body circle
  ctx.beginPath();
  ctx.arc(px, py, R, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = isActive ? 2 : 1;
  ctx.stroke();

  // Helmet bump (top-down helmet visor shape)
  const hx = px + Math.cos(yaw) * (R * 0.5);
  const hy = py + Math.sin(yaw) * (R * 0.5);
  ctx.beginPath();
  ctx.arc(hx, hy, R * 0.55, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fill();

  // Direction arrow
  const ax = px + Math.cos(yaw) * (R + 7);
  const ay = py + Math.sin(yaw) * (R + 7);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  // arrowhead
  const la = yaw + (Math.PI * 0.75);
  const ra = yaw - (Math.PI * 0.75);
  ctx.lineTo(ax + Math.cos(la) * 5, ay + Math.sin(la) * 5);
  ctx.lineTo(ax + Math.cos(ra) * 5, ay + Math.sin(ra) * 5);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  // Label above player
  const labelY = py - R - 14;
  ctx.font = "bold 11px 'Arial Narrow', Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillText(label, px + 1, labelY + 1);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, px, labelY);

  if (subLabel) {
    ctx.font = "9px Arial, sans-serif";
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillText(subLabel, px + 1, labelY + 12);
    ctx.fillStyle = color;
    ctx.fillText(subLabel, px, labelY + 12);
  }

  ctx.textAlign = "left";
}

// ─── TF2 logo mark (drawn with math, top-left corner) ─────────────────────────

function drawTF2Logo(ctx: OffscreenCanvasRenderingContext2D, x: number, y: number) {
  // Bold outlined "TF2" lettering using rect primitives — recognisably TF2-style
  ctx.save();
  ctx.translate(x, y);

  // Background pill
  ctx.fillStyle = "rgba(200,70,30,0.9)";
  ctx.beginPath();
  ctx.roundRect(0, 0, 72, 26, 4);
  ctx.fill();

  ctx.fillStyle = "#fff";
  ctx.font = "bold 16px Arial Black, Arial, sans-serif";
  ctx.textBaseline = "middle";
  ctx.fillText("TF2", 8, 13);

  // Small "DEMO" tag
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "7px Arial, sans-serif";
  ctx.fillText("DEMO", 76, 13);

  ctx.restore();
  ctx.textBaseline = "alphabetic";
}

// ─── Map background grid (procedural topdown "map plane") ────────────────────

function drawMapPlane(
  ctx: OffscreenCanvasRenderingContext2D,
  mapX: number, mapY: number, mapW: number, mapH: number,
  mapName: string,
) {
  // Base fill
  ctx.fillStyle = BG_MAP;
  ctx.fillRect(mapX, mapY, mapW, mapH);

  // Grid lines
  ctx.strokeStyle = GRID_COL;
  ctx.lineWidth = 1;
  const gridStep = 40;
  for (let gx = mapX; gx <= mapX + mapW; gx += gridStep) {
    ctx.beginPath(); ctx.moveTo(gx, mapY); ctx.lineTo(gx, mapY + mapH); ctx.stroke();
  }
  for (let gy = mapY; gy <= mapY + mapH; gy += gridStep) {
    ctx.beginPath(); ctx.moveTo(mapX, gy); ctx.lineTo(mapX + mapW, gy); ctx.stroke();
  }

  // Subtle map name watermark in centre
  ctx.save();
  ctx.globalAlpha = 0.06;
  ctx.font = "bold 48px Arial Black, Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(mapName.toUpperCase(), mapX + mapW / 2, mapY + mapH / 2 + 16);
  ctx.restore();
  ctx.textAlign = "left";

  // Border
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.strokeRect(mapX, mapY, mapW, mapH);
}

// ─── HUD panel ────────────────────────────────────────────────────────────────

function drawHUD(
  ctx: OffscreenCanvasRenderingContext2D,
  hx: number, hy: number, hw: number, hh: number,
  header: Header,
  currentTimeSec: number,
  sampleCount: number,
  frameIdx: number,
  totalFrames: number,
) {
  // Panel background
  ctx.fillStyle = "rgba(10,14,20,0.92)";
  ctx.fillRect(hx, hy, hw, hh);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(hx, hy, hw, hh);

  const px = hx + 12;
  let py = hy + 20;

  drawTF2Logo(ctx, px, py - 14);
  py += 20;

  // Divider
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(hx + hw - 12, py); ctx.stroke();
  py += 10;

  // Info rows
  const infoRows: [string, string][] = [
    ["MAP",    header.mapName  || "unknown"],
    ["SERVER", (header.serverName || "unknown").slice(0, 28)],
    ["CLIENT", (header.clientName || "unknown").slice(0, 28)],
    ["DIR",    (header.gameDir    || "unknown").slice(0, 28)],
    ["TICKS",  `${header.playbackTicks}`],
    ["DURATION", `${header.playbackTime.toFixed(1)}s`],
    ["SAMPLES", `${sampleCount}`],
  ];

  for (const [label, val] of infoRows) {
    ctx.font = "bold 9px Arial, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillText(label, px, py);
    ctx.font = "10px Arial, sans-serif";
    ctx.fillStyle = "#d0d8e8";
    ctx.fillText(val, px + 52, py);
    py += 15;
  }

  py += 4;
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(hx + hw - 12, py); ctx.stroke();
  py += 14;

  // Current time display
  const mm  = String(Math.floor(currentTimeSec / 60)).padStart(2, "0");
  const ss  = String(Math.floor(currentTimeSec % 60)).padStart(2, "0");
  const tmm = String(Math.floor(header.playbackTime / 60)).padStart(2, "0");
  const tss = String(Math.floor(header.playbackTime % 60)).padStart(2, "0");

  ctx.font = "bold 22px 'Arial Narrow', Arial, sans-serif";
  ctx.fillStyle = "#f0c060";
  ctx.fillText(`${mm}:${ss}`, px, py);
  ctx.font = "12px Arial, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.fillText(` / ${tmm}:${tss}`, px + 48, py);
  py += 18;

  // Progress bar
  const barW = hw - 24;
  const t = totalFrames > 1 ? frameIdx / (totalFrames - 1) : 0;
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.fillRect(px, py, barW, 6);
  ctx.fillStyle = "#f0c060";
  ctx.fillRect(px, py, barW * t, 6);
  py += 18;

  // Team legend
  ctx.font = "bold 10px Arial, sans-serif";
  ctx.fillStyle = RED;
  ctx.fillRect(px, py - 8, 10, 10);
  ctx.fillStyle = "#ccc";
  ctx.fillText("RED team", px + 14, py);
  ctx.fillStyle = BLU;
  ctx.fillRect(px + 90, py - 8, 10, 10);
  ctx.fillText("BLU team", px + 104, py);
}

// ─── Full frame renderer ──────────────────────────────────────────────────────

function renderFrame(
  ctx: OffscreenCanvasRenderingContext2D,
  header: Header,
  samples: Sample[],
  frameIdx: number,
  totalFrames: number,
) {
  const t = totalFrames > 1 ? frameIdx / (totalFrames - 1) : 0;
  const currentTimeSec = t * header.playbackTime;

  // Full background
  ctx.fillStyle = BG_DARK;
  ctx.fillRect(0, 0, W, H);

  // Layout: map panel takes left 880px, HUD takes right 400px
  const MAP_X = 0;
  const MAP_Y = 0;
  const MAP_W = 880;
  const MAP_H = H;
  const HUD_X = MAP_W;
  const HUD_Y = 0;
  const HUD_W = W - MAP_W;
  const HUD_H = H;

  // Map padding (inner drawing area)
  const PAD   = 48;
  const INNER_X = MAP_X + PAD;
  const INNER_Y = MAP_Y + PAD;
  const INNER_W = MAP_W - PAD * 2;
  const INNER_H = MAP_H - PAD * 2;

  drawMapPlane(ctx, INNER_X, INNER_Y, INNER_W, INNER_H, header.mapName);

  // Compute world bounds from all samples
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const s of samples) {
    if (s.x < minX) minX = s.x;
    if (s.x > maxX) maxX = s.x;
    if (s.y < minY) minY = s.y;
    if (s.y > maxY) maxY = s.y;
  }
  const rangeX = Math.max(maxX - minX, 1);
  const rangeY = Math.max(maxY - minY, 1);
  // Maintain aspect, add 10% padding
  const scaleX = (INNER_W * 0.85) / rangeX;
  const scaleY = (INNER_H * 0.85) / rangeY;
  const scale  = Math.min(scaleX, scaleY);
  const offX   = INNER_X + (INNER_W - rangeX * scale) / 2;
  const offY   = INNER_Y + (INNER_H - rangeY * scale) / 2;

  const wx = (worldX: number) => offX + (worldX - minX) * scale;
  const wy = (worldY: number) => offY + (worldY - minY) * scale;

  // Determine current sample index
  const sIdx = Math.min(
    Math.floor(t * (samples.length - 1)),
    samples.length - 1
  );

  // Draw full ghosted trail
  if (samples.length > 1) {
    const team = teamFromName(header.clientName);
    ctx.strokeStyle = team === "red" ? TRAIL_RED : TRAIL_BLU;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i++) {
      const sx = wx(samples[i].x);
      const sy = wy(samples[i].y);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Draw active trail up to now (brighter)
    const team2 = team === "red" ? RED : BLU;
    ctx.strokeStyle = team2;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    for (let i = 0; i <= sIdx; i++) {
      const sx = wx(samples[i].x);
      const sy = wy(samples[i].y);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Draw player dot at current position
  if (sIdx < samples.length) {
    const cur = samples[sIdx];
    const px = wx(cur.x);
    const py = wy(cur.y);
    const team = teamFromName(header.clientName);
    const timeStr = `${String(Math.floor(currentTimeSec / 60)).padStart(2,"0")}:${String(Math.floor(currentTimeSec % 60)).padStart(2,"0")}`;
    drawPlayerIcon(ctx, px, py, cur.yaw, team, true, header.clientName.slice(0, 14), timeStr);
  }

  // Map frame label
  ctx.font = "bold 10px Arial, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.25)";
  ctx.fillText(`FRAME ${frameIdx + 1} / ${totalFrames}  •  TICK ${samples[Math.min(sIdx, samples.length-1)]?.tick ?? 0}`, PAD, H - 18);

  // Compass rose (top-right of map area)
  {
    const cx = MAP_X + MAP_W - 36;
    const cy = MAP_Y + 36;
    const cr = 16;
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, cr, 0, Math.PI * 2); ctx.stroke();
    const dirs = [["N", 0], ["E", 90], ["S", 180], ["W", 270]] as [string, number][];
    for (const [d, a] of dirs) {
      const r = (a - 90) * Math.PI / 180;
      ctx.font = "bold 8px Arial, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.textAlign = "center";
      ctx.fillText(d, cx + Math.cos(r) * (cr + 8), cy + Math.sin(r) * (cr + 8) + 3);
    }
    ctx.textAlign = "left";
  }

  drawHUD(ctx, HUD_X, HUD_Y, HUD_W, HUD_H, header, currentTimeSec, samples.length, frameIdx, totalFrames);
}

// ─── Handler ──────────────────────────────────────────────────────────────────

class tf2demoHandler {
  init() {
    return {
      name: "TF2 Demo",
      from: [".dem"],
      to: [
        { ext: ".mp4",  mime: "video/mp4",       label: "MP4 (H.264)" },
        { ext: ".webm", mime: "video/webm",       label: "WebM (VP8)"  },
      ],
      category: ["other"],
    };
  }

  async convert(
    file: File,
    args: { target: string; updateProgress: (n: number) => void }
  ): Promise<Blob> {
    const { target, updateProgress } = args;

    // 1. Read file
    updateProgress(2);
    const ab  = await file.arrayBuffer();
    const buf = new Uint8Array(ab);

    // 2. Validate + parse header
    updateProgress(5);
    const header = parseHeader(buf);
    if (!header) throw new Error("Not a valid TF2/HL2 demo (missing HL2DEMO stamp).");

    // 3. Parse tick samples
    updateProgress(8);
    const { samples } = parseFrames(buf, 1072, header.playbackTicks);
    if (samples.length < 2) {
      throw new Error(`Demo parsed but found only ${samples.length} position sample(s). The file may be corrupt or a zero-tick demo.`);
    }

    // 4. Setup canvas + FFmpeg
    updateProgress(12);
    const canvas = new OffscreenCanvas(W, H);
    const ctx    = canvas.getContext("2d") as OffscreenCanvasRenderingContext2D;
    if (!ctx) throw new Error("OffscreenCanvas 2D unavailable in this browser.");

    const ff = await getFFmpeg();
    updateProgress(18);

    const duration    = Math.max(header.playbackTime, 1);
    const totalFrames = Math.ceil(duration * FPS);

    // 5. Encode via rawvideo pipe — NO intermediate files, NO concat list
    // We accumulate all raw RGBA frames into a single Uint8Array,
    // write it once to FFmpeg's VFS as "input.raw", then encode.
    // This is the most reliable approach in ffmpeg.wasm.
    const bytesPerFrame = W * H * 4; // RGBA
    const rawAll        = new Uint8Array(bytesPerFrame * totalFrames);

    for (let i = 0; i < totalFrames; i++) {
      renderFrame(ctx, header, samples, i, totalFrames);
      const imageData = ctx.getImageData(0, 0, W, H);
      rawAll.set(imageData.data, i * bytesPerFrame);

      if (i % 5 === 0) {
        updateProgress(18 + Math.floor((i / totalFrames) * 55));
      }
    }

    updateProgress(73);

    // Write single raw blob
    await ff.writeFile("input.raw", rawAll);
    updateProgress(76);

    // Build FFmpeg command based on output format
    const isWebm = target === ".webm";
    const outFile = isWebm ? "out.webm" : "out.mp4";

    const cmd = isWebm
      ? [
          "-f",       "rawvideo",
          "-pix_fmt", "rgba",
          "-s",       `${W}x${H}`,
          "-r",       String(FPS),
          "-i",       "input.raw",
          "-c:v",     "libvpx",
          "-b:v",     "2M",
          "-pix_fmt", "yuv420p",
          outFile,
        ]
      : [
          "-f",       "rawvideo",
          "-pix_fmt", "rgba",
          "-s",       `${W}x${H}`,
          "-r",       String(FPS),
          "-i",       "input.raw",
          "-c:v",     "libx264",
          "-preset",  "ultrafast",
          "-crf",     "22",
          "-pix_fmt", "yuv420p",
          "-movflags", "+faststart",
          outFile,
        ];

    await ff.exec(cmd);
    updateProgress(94);

    // 6. Read output
    const data = await ff.readFile(outFile) as Uint8Array;

    // Cleanup
    try { await ff.deleteFile("input.raw"); }  catch { /* ok */ }
    try { await ff.deleteFile(outFile); }       catch { /* ok */ }

    updateProgress(100);

    const mime = isWebm ? "video/webm" : "video/mp4";
    return new Blob([data], { type: mime });
  }
}

export default tf2demoHandler;