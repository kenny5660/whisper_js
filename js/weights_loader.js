import h5wasm from "h5wasm";
import * as tf from '@tensorflow/tfjs';

export class Weights {

    static _MODELS = {
        "tiny.en": "",
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
        this.model_name = model_name;
    }

    async init_weights() {
        const { FS } = await h5wasm.ready;
        
        let model_url = Weights._MODELS[this.model_name];

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