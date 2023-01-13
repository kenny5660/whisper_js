import * as tf from '@tensorflow/tfjs';


class BeamSearchDecoderLayer extends tf.layers.Layer {
    constructor(kwargs) {
        super(kwargs);
        this.beamWidth = kwargs.beamWidth;
        this.topPaths = kwargs.topPaths;
    }

    static get className() {
        return 'BeamSearchDecoderLayer';
    }

    call(inputs, kwargs) {
        const logits = inputs;
        const decoder = new BeamSearchDecoderLayer(kwargs);
        for (let i = 0; i < logits[0].length; i++) {
            decoder.addHypothesis([i], logits[0][i]);
        }
        for (let t = 1; t < logits.length; t++) {
            const newHypotheses = [];
            const newProbs = [];
            for (let i = 0; i < decoder.getHypotheses().length; i++) {
                const hyp = decoder.getHypotheses()[i];
                const prob = decoder.getProbabilities()[i];
                for (let j = 0; j < logits[t].length; j++) {
                    newHypotheses.push(hyp.concat(j));
                    newProbs.push(prob * logits[t][j]);
                }
            }
            for (let i = 0; i < newHypotheses.length; i++) {
                decoder.addHypothesis(newHypotheses[i], newProbs[i]);
            }
        }
        return decoder.getHypotheses();
    }

    getProbabilities() {

    }

    getHypotheses() {

    }

    addHypothesis(numbers, logitElement) {

    }
}