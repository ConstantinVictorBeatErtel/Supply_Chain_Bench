import { sha256Bytes } from "./sha256.js";

const PROTOCOL_ID = "live-y-domain-randomized-grpo-v1";

export function researchDigest(seedHex, namespace, index = 0) {
  if (!Number.isInteger(index) || index < 0) throw new RangeError("draw index must be non-negative");
  return sha256Bytes(`${PROTOCOL_ID}|${seedHex}|${namespace}|${index}`);
}

export function researchUniform(seedHex, namespace, index = 0) {
  const bytes = researchDigest(seedHex, namespace, index);
  let value = 0n;
  for (let offset = 0; offset < 8; offset += 1) {
    value = (value << 8n) | BigInt(bytes[offset]);
  }
  return Number(value >> 11n) / 2 ** 53;
}

export function researchPoisson(seedHex, namespace, lambda, index = 0) {
  const rate = Number(lambda);
  if (!Number.isFinite(rate) || rate < 0) throw new RangeError("lambda must be finite and non-negative");
  if (rate === 0) return 0;
  const threshold = Math.exp(-rate);
  let product = 1;
  let count = 0;
  while (product > threshold) {
    product *= researchUniform(seedHex, `${namespace}/poisson`, count + index * 4096);
    count += 1;
  }
  return count - 1;
}

export function negativeBinomialR10P05(seedHex, namespace, index = 0) {
  let total = 0;
  let successes = 0;
  while (successes < 10) {
    const draw = researchUniform(seedHex, `${namespace}/geometric`, index + total + successes * 4096);
    if (draw >= 0.5) successes += 1;
    else total += 1;
  }
  return total;
}
