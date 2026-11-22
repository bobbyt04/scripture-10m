// Reads the binary written by web/scripts/export_model.py (layout documented there).

export interface ModelConfig {
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
}

interface TensorEntry {
  name: string;
  shape: number[];
  dtype: "float32" | "float16" | "int8";
  offset: number;
  scale?: number;
}

export interface ModelWeights {
  format: string;
  config: ModelConfig;
  /** Every tensor dequantised to float32, keyed by its PyTorch state_dict name. */
  tensors: Map<string, Float32Array>;
}

let halfTable: Float32Array | null = null;

/** IEEE half -> float32, exact, via a 65,536-entry table. */
function halfToFloatTable(): Float32Array {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    let v: number;
    if (exp === 0) v = frac * 2 ** -24;
    else if (exp === 31) v = frac ? NaN : Infinity;
    else v = (1 + frac / 1024) * 2 ** (exp - 15);
    halfTable[h] = sign * v;
  }
  return halfTable;
}

export function parseModel(buf: ArrayBuffer): ModelWeights {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== "S10M") throw new Error("not a Scripture-10M model file");
  const view = new DataView(buf);
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported model file version ${version}`);
  const headerLen = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLen))) as {
    format: string;
    config: ModelConfig;
    tensors: TensorEntry[];
  };
  const base = 12 + headerLen;
  const tensors = new Map<string, Float32Array>();
  for (const t of header.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const start = base + t.offset;
    const out = new Float32Array(n);
    if (t.dtype === "float32") {
      out.set(new Float32Array(buf.slice(start, start + 4 * n)));
    } else if (t.dtype === "float16") {
      const table = halfToFloatTable();
      const src = new Uint16Array(buf.slice(start, start + 2 * n));
      for (let i = 0; i < n; i++) out[i] = table[src[i]];
    } else {
      const src = new Int8Array(buf, start, n);
      const scale = Math.fround(t.scale!);
      for (let i = 0; i < n; i++) out[i] = src[i] * scale; // same rounding as q.float() * scale in PyTorch
    }
    tensors.set(t.name, out);
  }
  return { format: header.format, config: header.config, tensors };
}
