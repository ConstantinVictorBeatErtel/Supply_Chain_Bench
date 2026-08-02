// CPython's _random.Random integer seeding, random(), and gauss() behavior.
// This is the MT19937 implementation used by Python 3.11+.

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

export class PythonRandom {
  constructor(seed) {
    this.state = new Uint32Array(N);
    this.index = N;
    this.gaussNext = null;
    this.seed(BigInt(seed));
  }

  seed(seed) {
    let value = seed < 0n ? -seed : seed;
    const key = [];
    do {
      key.push(Number(value & 0xffffffffn) >>> 0);
      value >>= 32n;
    } while (value > 0n);
    this.initByArray(key);
    this.gaussNext = null;
  }

  initGenRand(seed) {
    this.state[0] = seed >>> 0;
    for (let index = 1; index < N; index += 1) {
      const previous = this.state[index - 1];
      this.state[index] = (
        Math.imul(1812433253, previous ^ (previous >>> 30)) + index
      ) >>> 0;
    }
    this.index = N;
  }

  initByArray(key) {
    this.initGenRand(19650218);
    let i = 1;
    let j = 0;
    let count = Math.max(N, key.length);
    for (; count > 0; count -= 1) {
      const previous = this.state[i - 1];
      this.state[i] = (
        (this.state[i] ^ Math.imul(previous ^ (previous >>> 30), 1664525))
        + key[j]
        + j
      ) >>> 0;
      i += 1;
      j += 1;
      if (i >= N) {
        this.state[0] = this.state[N - 1];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (count = N - 1; count > 0; count -= 1) {
      const previous = this.state[i - 1];
      this.state[i] = (
        (this.state[i] ^ Math.imul(previous ^ (previous >>> 30), 1566083941))
        - i
      ) >>> 0;
      i += 1;
      if (i >= N) {
        this.state[0] = this.state[N - 1];
        i = 1;
      }
    }
    this.state[0] = 0x80000000;
  }

  genUint32() {
    if (this.index >= N) {
      let y;
      let index = 0;
      for (; index < N - M; index += 1) {
        y = (this.state[index] & UPPER_MASK) | (this.state[index + 1] & LOWER_MASK);
        this.state[index] = this.state[index + M] ^ (y >>> 1) ^ ((y & 1) ? MATRIX_A : 0);
      }
      for (; index < N - 1; index += 1) {
        y = (this.state[index] & UPPER_MASK) | (this.state[index + 1] & LOWER_MASK);
        this.state[index] = this.state[index + (M - N)] ^ (y >>> 1) ^ ((y & 1) ? MATRIX_A : 0);
      }
      y = (this.state[N - 1] & UPPER_MASK) | (this.state[0] & LOWER_MASK);
      this.state[N - 1] = this.state[M - 1] ^ (y >>> 1) ^ ((y & 1) ? MATRIX_A : 0);
      this.index = 0;
    }
    let value = this.state[this.index];
    this.index += 1;
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c5680;
    value ^= (value << 15) & 0xefc60000;
    value ^= value >>> 18;
    return value >>> 0;
  }

  random() {
    const a = this.genUint32() >>> 5;
    const b = this.genUint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  gauss(mu = 0, sigma = 1) {
    let z = this.gaussNext;
    this.gaussNext = null;
    if (z === null) {
      const x2pi = this.random() * (2 * Math.PI);
      const g2rad = Math.sqrt(-2 * Math.log(1 - this.random()));
      z = Math.cos(x2pi) * g2rad;
      this.gaussNext = Math.sin(x2pi) * g2rad;
    }
    return mu + z * sigma;
  }

  snapshot() {
    return {
      index: this.index,
      gaussNext: this.gaussNext,
      state: Array.from(this.state),
    };
  }
}
