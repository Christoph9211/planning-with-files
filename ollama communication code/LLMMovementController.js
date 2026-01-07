class LLMMovementController {
  /**
   * Constructs a new LLMMovementController instance.
   *
   * @param {Object} [options] - Options to use for the constructor.
   * @param {string} [options.apiUrl='http://localhost:11434/api/generate'] - The URL of the API endpoint to use for generating
   *   movement patterns.
   * @param {string} [options.model='hf.co/mradermacher/Abliterated-Emo_Alice-3.2-1B-i1-GGUF:Q4_K_M'] - The model to use for generating movement patterns.
   */
  constructor({ apiUrl = 'http://localhost:11434/api/generate', model = 'hf.co/mradermacher/Abliterated-Emo_Alice-3.2-1B-i1-GGUF:Q4_K_M' } = {}) {
    this.apiUrl = apiUrl;
    this.model = model;
  }

  /**
   * Generates a prompt to send to the LLM to generate a sequence of motor
   * commands based on the given description and optional currentState object.
   * @param {string} description - A string describing the movement pattern to generate
   * @param {number} durationMs - Use duration to determine complexity of phases
   * @param {Object} [currentState] - An object describing the current state of the
   *   movement pattern, which is included in the prompt to the LLM.
   * @returns {string} A string prompt to send to the LLM.
   */
  async generateMovementPrompt(description, durationMs, currentState) {
    const state = JSON.stringify(currentState || {});
    // Extract last value if available for continuity context
    const lastValue = currentState?.previous && currentState.previous.length
      ? currentState.previous[currentState.previous.length - 1]
      : 0.0;

    const isShort = durationMs < 2000;

    return [
      `You are a movement composer for a linear actuator grid.`,
      `Goal: ${description || 'Compose a dynamic, natural-feeling sequence.'}`,
      `Context: Start intensity is ${lastValue.toFixed(2)}. Duration is ${durationMs}ms.`,
      `Additional State: ${state}`,
      `Return ONLY minified JSON with this structure:`,
      `{"phases":[`,
      isShort
        ? `  {"name":"main","duration_ratio":1.0,"floor":0.1,"peak":0.9,"intensity_curve":"easeInOut","tempo_hz":0.5,"variation":0.2,"rests":0.0}`
        : `  {"name":"intro","duration_ratio":0.15,"floor":0.05,"peak":0.35,"intensity_curve":"easeIn","tempo_hz":0.3,"variation":0.15,"rests":0.05}, ... (more phases)`,
      `],"micro":{"humanize_jitter":0.05,"swing":0.08},"seed":null}`,
      `- duration_ratio values must sum to 1.0.`,
      `- Use simple curve names: easeIn|easeOut|easeInOut|sine|surge|decay.`,
      `- Keep numbers between 0 and 1.`,
      `- Start the first phase near the Context Start intensity if possible.`,
      `- Do not include any commentary, just the JSON.`
    ].join('\n');
  }

  /**
   * Generates a movement sequence based on the given description and
   * duration. Optionally takes a currentState object which is included in
   * the prompt sent to the LLM.
   * @param {string} description - A string describing the movement pattern to generate.
   * @param {number} duration - The duration of the pattern in ms.
   * @param {Object} currentState - An optional object with information about the current state
   *   of the device. This is included in the prompt sent to the LLM.
   * @returns {number[]} A 1D array of numbers representing the movement pattern.
   *   The values should be in the range [0, 1] and represent the intensity of the
   *   movement at each point in time.
   */
  async generateMovementSequence(description, duration, currentState = {}) {
    const prompt = await this.generateMovementPrompt(description, duration, currentState);
    let startValue = 0;
    if (currentState.previous && currentState.previous.length > 0) {
      startValue = currentState.previous[currentState.previous.length - 1];
    }

    try {
      const body = {
        model: this.model,
        prompt,
        stream: false,
        format: 'json',
        options: { temperature: 0.7 }
      };
      const res = await fetch(this.apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      const text = data.response || data.generated_text || '';
      const spec = this.safeParseSpec(text);

      if (spec && Array.isArray(spec.phases) && spec.phases.length) {
        return this.synthesizeFromPhases(spec, duration, startValue);
      }

      // Fallback
      const movementParams = this.parseMovementOutput(text);
      return this.convertToActuatorCommands(movementParams, duration, startValue);
    } catch (error) {
      console.error('Error generating movement:', error);
      return this.generateFallbackPattern(duration, startValue);
    }
  }

  // --- Parsing & synthesis helpers ---

  safeParseSpec(text) {
    if (!text) return null;
    try {
      const match = text.match(/\{[\s\S]*\}/);
      const json = match ? match[0] : text;
      return JSON.parse(json);
    } catch (_) {
      return null;
    }
  }

  parseMovementOutput(text) {
    const params = {
      intensity: 0.5,
      frequency: 1.0,
      smoothness: 0.8,
      variation: 0.3
    };
    if (text.includes('gentle')) params.intensity *= 0.5;
    if (text.includes('intense')) params.intensity *= 1.5;
    if (text.includes('slow')) params.frequency *= 0.5;
    if (text.includes('fast')) params.frequency *= 1.5;
    if (text.includes('smooth')) params.smoothness = 1.0;
    if (text.includes('erratic')) params.variation = 0.8;
    return params;
  }

  convertToActuatorCommands(params, duration, startValue = 0) {
    const steps = Math.max(1, Math.floor(duration / 100));
    const commands = [];
    for (let i = 0; i < steps; i++) {
      const time = i / steps;
      const baseIntensity = params.intensity;
      const variation = Math.sin(time * Math.PI * 2 * params.frequency) * params.variation;
      const smoothness = Math.sin(time * Math.PI * 2) * (1 - params.smoothness);
      const raw = baseIntensity + variation + smoothness;
      // Blend from startValue
      const blend = i < 5 ? (i / 5) : 1;
      const val = (startValue * (1 - blend)) + (raw * blend);
      commands.push(this._clamp01(val));
    }
    return this._postProcess(commands, { maxDelta: 0.12, smoothAlpha: 0.35 });
  }

  /**
   * Synthesizes a movement pattern from a specification.
   * @param {Object} spec - The specification object
   * @param {number} duration - The duration of the pattern in ms
   * @param {number} startValue - The value to blend from at the start
   */
  synthesizeFromPhases(spec, duration, startValue = 0) {
    const steps = Math.max(1, Math.floor(duration / 100));
    const phases = this._normalizePhases(spec.phases);
    const jitter = spec.micro?.humanize_jitter ?? 0.03;
    const swing = spec.micro?.swing ?? 0.0;
    const seed = Number.isFinite(spec.seed) ? spec.seed : Math.floor(Math.random() * 1e9);
    const rand = this._mulberry32(seed);

    const curve = new Array(steps).fill(0);
    let offset = 0;
    const stepSeconds = 0.1;

    for (const p of phases) {
      const count = Math.max(1, Math.floor(steps * p.duration_ratio));
      for (let i = 0; i < count && offset + i < steps; i++) {
        const t = i / Math.max(1, count - 1);
        const shaped = this._applyCurve(t, p.intensity_curve);
        const base = this._lerp(p.floor, p.peak, shaped);
        // tempo modulation
        const timeSec = (offset + i) * stepSeconds;
        const tempoMod = Math.sin(2 * Math.PI * (p.tempo_hz || 0.5) * timeSec) * (p.variation || 0) * 0.5;
        curve[offset + i] = this._clamp01(base + tempoMod);
      }
      offset += count;
    }
    for (let i = offset; i < steps; i++) curve[i] = curve[i - 1] ?? 0.1;

    // Add noise
    const noise = this._fbmNoise(steps, rand, 3);
    for (let i = 0; i < steps; i++) {
      const p = this._phaseAtIndex(phases, i / steps);
      const v = (p?.variation ?? 0.2);
      curve[i] = this._clamp01(curve[i] + (noise[i] - 0.5) * 2 * v * 0.15);
    }

    // Add rests
    for (let i = 0; i < steps; i++) {
      const p = this._phaseAtIndex(phases, i / steps);
      const restProb = p?.rests ?? 0;
      if (rand() < restProb * 0.2) {
        curve[i] = this._lerp(p.floor ?? 0.05, curve[i], 0.2);
      }
    }

    // Micro-jitter & swing
    for (let i = 0; i < steps; i++) {
      const j = (rand() - 0.5) * 2 * jitter;
      const s = (i % 2 === 0 ? -1 : 1) * swing * 0.05;
      curve[i] = this._clamp01(curve[i] + j + s);
    }

    // Continuity blending: Blend first few steps from startValue
    const blendSteps = Math.min(5, steps);
    for (let i = 0; i < blendSteps; i++) {
      const factor = i / blendSteps; // 0 to 1
      curve[i] = this._lerp(startValue, curve[i], factor);
    }

    // Improvements: Use slightly tighter physics constraint
    return this._postProcess(curve, { maxDelta: 0.15, smoothAlpha: 0.2 });
  }

  _normalizePhases(phases) {
    const sanitized = phases.map(p => ({
      name: String(p.name || ''),
      duration_ratio: this._clamp01(Number(p.duration_ratio ?? 0.2)),
      floor: this._clamp01(Number(p.floor ?? 0.1)),
      peak: this._clamp01(Number(p.peak ?? 0.8)),
      intensity_curve: String(p.intensity_curve || 'easeInOut'),
      tempo_hz: Math.max(0, Number(p.tempo_hz ?? 0.5)),
      variation: this._clamp01(Number(p.variation ?? 0.2)),
      rests: this._clamp01(Number(p.rests ?? 0.05)),
    }));
    let sum = sanitized.reduce((a, b) => a + b.duration_ratio, 0);
    if (sum <= 0) {
      return [{ name: 'main', duration_ratio: 1, floor: 0.1, peak: 0.8, intensity_curve: 'easeInOut', tempo_hz: 0.5, variation: 0.2, rests: 0.05 }];
    }
    return sanitized.map(p => ({ ...p, duration_ratio: p.duration_ratio / sum }));
  }

  _phaseAtIndex(phases, t) {
    let acc = 0;
    for (const p of phases) {
      const next = acc + p.duration_ratio;
      if (t <= next + 1e-6) return p;
      acc = next;
    }
    return phases[phases.length - 1];
  }

  _applyCurve(t, name) {
    const x = this._clamp01(t);
    switch ((name || '').toLowerCase()) {
      case 'easein': return x * x;
      case 'easeout': return 1 - (1 - x) * (1 - x);
      case 'sine': return 0.5 - 0.5 * Math.cos(Math.PI * x);
      case 'surge': return Math.pow(x, 0.6) * (1 + 0.1 * Math.sin(10 * x));
      case 'decay': return 1 - Math.pow(1 - x, 0.6);
      case 'easeinout':
      default: return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
    }
  }

  /**
   * Applies physics-like properties: limit abrupt velocity changes and smooth momentum.
   */
  _postProcess(arr, { maxDelta = 0.2, smoothAlpha = 0.3 } = {}) {
    const res = arr.slice();
    // 1. Velocity Clamp
    for (let i = 1; i < res.length; i++) {
      let diff = res[i] - res[i - 1];
      if (Math.abs(diff) > maxDelta) {
        diff = Math.sign(diff) * maxDelta;
        res[i] = res[i - 1] + diff;
      }
    }
    // 2. Momentum / Smoothing (EMA)
    const smoothed = [];
    let val = res[0];
    for (let i = 0; i < res.length; i++) {
      val = val * (1 - smoothAlpha) + res[i] * smoothAlpha;
      smoothed[i] = this._clamp01(val);
    }
    return smoothed;
  }

  _lerp(a, b, t) { return a + (b - a) * t; }
  _clamp01(x) { return Math.max(0, Math.min(1, x)); }

  _mulberry32(seed) {
    let t = (seed >>> 0) + 0x6D2B79F5;
    return function () {
      t |= 0; t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  _fbmNoise(n, rand, octaves = 3) {
    const base = new Array(n).fill(0).map(() => rand());
    const out = new Array(n).fill(0);
    let amp = 1, sumAmp = 0;
    for (let o = 0; o < octaves; o++) {
      const step = Math.max(1, Math.floor(n / Math.pow(2, o + 2)));
      const smooth = this._smoothArray(base, step);
      for (let i = 0; i < n; i++) out[i] += smooth[i] * amp;
      sumAmp += amp;
      amp *= 0.5;
    }
    for (let i = 0; i < n; i++) out[i] = out[i] / (sumAmp || 1);
    return out;
  }

  _smoothArray(arr, windowSize) {
    const out = new Array(arr.length).fill(0);
    const w = Math.max(1, windowSize | 0);
    let acc = 0;
    for (let i = 0; i < arr.length; i++) {
      acc += arr[i];
      if (i >= w) acc -= arr[i - w];
      out[i] = acc / Math.min(i + 1, w);
    }
    return out;
  }

  generateFallbackPattern(duration, startValue = 0.5) {
    const steps = Math.floor(duration / 100);
    // Simple sine fade from startValue
    return Array.from({ length: steps }, (_, i) => {
      const t = i / steps;
      // animate somewhat
      return this._clamp01(startValue * (1 - t) + 0.5 * t + 0.1 * Math.sin(t * 10));
    });
  }
}

export default LLMMovementController;
