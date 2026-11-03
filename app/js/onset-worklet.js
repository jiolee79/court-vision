import { OnsetCore } from './onset-core.js';

class OnsetProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.core = new OnsetCore(sampleRate);
    this.port.onmessage = (e) => {
      if (e.data.sensitivity != null) this.core.setSensitivity(e.data.sensitivity);
    };
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      const s = this.core.process(ch);
      if (s) this.port.postMessage({ strength: s, time: currentTime });
    }
    return true;
  }
}

registerProcessor('onset-processor', OnsetProcessor);
