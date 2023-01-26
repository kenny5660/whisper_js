import h5wasm from "h5wasm";
import * as tf from '@tensorflow/tfjs';
export class Weights {
    constructor(model_name) {
        this.model_name = model_name;
        this.file_name = this.model_name + ".h5";
        this.weights_inited = false;
        this.weights = null;
    }

    async init_weights(arrayBuffer) {
        const { FS } = await h5wasm.ready;
        console.log(arrayBuffer)
        FS.writeFile(this.file_name, arrayBuffer);
        this.weights = new h5wasm.File(this.file_name, "r");
        if (!this.weights.get('model_state_dict')) {
            console.log("parse h5 file error")
            throw new Error("parse h5 file error")
        }
        this.weights_inited = true;
    }

    get(key) {
        let data_key = this.weights.get('model_state_dict').get(key);
        return tf.tensor(data_key.value, data_key.shape);
    }
}

