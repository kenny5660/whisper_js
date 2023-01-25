import h5wasm from "h5wasm";
import * as tf from '@tensorflow/tfjs';

class Weights {

    static _MODELS = {
        "tiny.en": "https://raw.githubusercontent.com/kenny5660/whisper_js/weights/tiny.en.h5",
        "tiny": "",
        "base.en": "",
        "base": "",
        "small.en": "",
        "small": "",
        "medium.en": "",
        "medium": "",
        "large": ""
    }
    
    constructor(model_name) {
        this.init_(model_name);
    }

    async init_(model_name) {
        const { FS } = await h5wasm.ready;

        let model_url = this._MODELS[model_name];

        let response = await fetch(model_url);
        let ab = await response.arrayBuffer();

        FS.writeFile("weights.h5", new Uint8Array(ab));

        this.weights = new h5wasm.File("weights.h5", "r");
    }

    get(key) {
        let data_key = this.weights.get('model_state_dict').get(key);
        return tf.tensor(data_key.value, data_key.shape);
    }

    static available_models() {
        return Object.keys(_MODELS);
    }
}