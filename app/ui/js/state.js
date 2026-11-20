(() => {
  'use strict';

  const app = (window.MaskVox = {});

  app.createStore = (initial) => {
    let state = initial;
    const listeners = new Set();
    return {
      read: () => state,
      update(patch) {
        const changed = Object.keys(patch).filter((key) => !Object.is(state[key], patch[key]));
        if (!changed.length) return;
        state = { ...state, ...patch };
        for (const listener of listeners) listener(state, changed);
      },
      subscribe(listener) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };
  };

  app.defaults = Object.freeze({
    input_id: null, output_id: null, monitor_id: null, selected_voice_id: null,
    pitch: 0, index_rate: 0.5, gate_db: -50, latency_mode: 'balanced',
    monitor: false, monitor_volume: 0.6, gpu_index: 0, onboarding_dismissed: false,
  });
})();
