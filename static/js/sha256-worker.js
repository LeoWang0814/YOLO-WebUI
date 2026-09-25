/* Incremental SHA-256 worker; keeps large uploads off the UI thread. */
(() => {
  const K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]);
  const initial = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));
  const compress = (state, chunk) => {
    const words = new Uint32Array(64);
    for (let i = 0; i < 16; i++) words[i] = (chunk[i * 4] << 24) | (chunk[i * 4 + 1] << 16) | (chunk[i * 4 + 2] << 8) | chunk[i * 4 + 3];
    for (let i = 16; i < 64; i++) {
      const x = words[i - 15];
      const y = words[i - 2];
      const s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >>> 3);
      const s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >>> 10);
      words[i] = (words[i - 16] + s0 + words[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = state;
    for (let i = 0; i < 64; i++) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + K[i] + words[i]) >>> 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) >>> 0;
      [h, g, f, e, d, c, b, a] = [g, f, e, (d + temp1) >>> 0, c, b, a, (temp1 + temp2) >>> 0];
    }
    state[0] = (state[0] + a) >>> 0; state[1] = (state[1] + b) >>> 0; state[2] = (state[2] + c) >>> 0; state[3] = (state[3] + d) >>> 0;
    state[4] = (state[4] + e) >>> 0; state[5] = (state[5] + f) >>> 0; state[6] = (state[6] + g) >>> 0; state[7] = (state[7] + h) >>> 0;
  };
  self.onmessage = async (event) => {
    const file = event.data.file;
    const chunkSize = Number(event.data.chunkSize) || 8 * 1024 * 1024;
    const state = initial.slice();
    let pending = new Uint8Array(0);
    let processed = 0;
    const consume = (bytes) => {
      if (!bytes.length) return;
      const joined = new Uint8Array(pending.length + bytes.length);
      joined.set(pending); joined.set(bytes, pending.length);
      let complete = joined.length - (joined.length % 64);
      for (let offset = 0; offset < complete; offset += 64) compress(state, joined.subarray(offset, offset + 64));
      pending = joined.slice(complete);
    };
    try {
      while (processed < file.size) {
        const bytes = new Uint8Array(await file.slice(processed, processed + chunkSize).arrayBuffer());
        consume(bytes);
        processed += bytes.length;
        self.postMessage({ type: "progress", processed, total: file.size });
      }
      const bitLength = file.size * 8;
      const finalLength = ((pending.length + 9 + 63) >> 6) * 64;
      const finalBlock = new Uint8Array(finalLength);
      finalBlock.set(pending); finalBlock[pending.length] = 0x80;
      const high = Math.floor(bitLength / 0x100000000);
      const low = bitLength >>> 0;
      finalBlock[finalLength - 8] = high >>> 24; finalBlock[finalLength - 7] = high >>> 16; finalBlock[finalLength - 6] = high >>> 8; finalBlock[finalLength - 5] = high;
      finalBlock[finalLength - 4] = low >>> 24; finalBlock[finalLength - 3] = low >>> 16; finalBlock[finalLength - 2] = low >>> 8; finalBlock[finalLength - 1] = low;
      for (let offset = 0; offset < finalBlock.length; offset += 64) compress(state, finalBlock.subarray(offset, offset + 64));
      self.postMessage({ type: "complete", sha256: state.map(value => value.toString(16).padStart(8, "0")).join("") });
    } catch (error) {
      self.postMessage({ type: "error", message: error?.message || "Unable to hash the file." });
    }
  };
})();
