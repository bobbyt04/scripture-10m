// Checks a downloaded model file against src/model/manifest.json before it is parsed: a cut-off or altered
// download fails with a clear error instead of producing nonsense.

export async function verifyDownload(data: ArrayBuffer, bytes: number, sha256: string): Promise<void> {
  if (data.byteLength !== bytes) {
    throw new Error(`model download has ${data.byteLength} bytes, expected ${bytes}`);
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== sha256) throw new Error("model download does not match its checksum");
}
