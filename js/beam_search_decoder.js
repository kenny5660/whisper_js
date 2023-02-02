tf = require('@tensorflow/tfjs');

/*
constructor:
- beamSize количество разных поисков, == 1 почти равно GreedyDecoder
- eot - символ конца последовательности
- inference - класс для кэширования
- patience - коэффициент уверенности
- maxCandidates - максимальное количество гипотез
- finishedSequences - итоговая последовательность токенов
 */

class BeamSearchDecoder {
    constructor(beamSize, eot, inference, patience = 1.0) {
        this.beamSize = beamSize;
        this.eot = eot;
        this.inference = inference;
        this.patience = patience;
        this.maxCandidates = Math.round(beamSize * this.patience);
        this.finishedSequences = null;

        if (this.maxCandidates <= 0) {
            throw new Error(`Invalid beam size (${beamSize}) or patience (${patience})`);
        }
    }

    reset() {
        this.finishedSequences = null;
    }

    update(tokens, logits, sumLogprobs) {
        if (tokens.shape[0] % this.beamSize !== 0) {
            throw new Error(`${tokens.shape[0]} % ${this.beamSize} !== 0`);
        }

        const nAudio = Math.floor(tokens.shape[0] / this.beamSize);

        if (this.finishedSequences === null) {
            this.finishedSequences = Array(nAudio).fill({});
        }

        let logProbs = tf.logSoftmax(logits, -1);
        let nextTokens = [];
        let sourceIndices = [];
        let finishedSequences = [];

        for (let i = 0; i < nAudio; i++) {
            const scores = {};
            const sources = {};
            const finished = {};

            for (let j = 0; j < this.beamSize; j++) {
                const idx = i * this.beamSize + j;
                const prefix = tokens.slice([idx], [1]).arraySync();

                const {logProbsTmp, indicesTmp} = tf.topk(logProbs[idx], this.beamSize + 1);

                for (let k = 0; k < indices.shape[0]; k++) {
                    let newLogprob = (sumLogprobs[idx].add(logProbsTmp[idx])).dataSync();
                    let sequence = prefix + [indicesTmp[idx].dataSync()];
                    scores[sequence] = newLogprob;
                    sources[sequence] = idx;
                }
            }

            let saved = 0;
            for (const sequence of Object.keys(scores).sort((a, b) => scores[b] - scores[a])) {
                if (sequence[sequence.length - 1] === this.eot) {
                    finished[sequence] = scores[sequence];
                }
                else {
                    sumLogprobs[nextTokens.shape[0]] = scores[sequence];
                    nextTokens.push(sequence);
                    sourceIndices.push(sources[sequence]);
                    saved++;
                    if (saved === this.beamSize){
                        break;
                    }
                }
            }

            finishedSequences.push(finished);
        }

        tokens = tf.tensor(nextTokens);
        this.inference.rearrangeKVCache(sourceIndices);

        for (let i = 0; i < this.finishedSequences.length; i++) {
            for (const seq of Object.keys(finishedSequences[i]).sort((a, b) => finishedSequences[i][b] - finishedSequences[i][a])) {
                if (Object.keys(this.finishedSequences[i]).length >= this.maxCandidates) {
                    break;
                }
                this.finishedSequences[i][seq] = finishedSequences[i][seq];
            }
        }

        const completed = this.finishedSequences.every(sequences => Object.keys(sequences).length >= this.maxCandidates);
        return [ tokens, completed ];
    }

    finalize(precedingTokens, sumLogprobs) {
        for (let i = 0; i < this.finishedSequences.length; i++) {
            let sequences = this.finishedSequences[i];
            if (sequences.length < this.beamSize){
                const sortedIndices = Array.from(sumLogprobs[i].argMin().dataSync()).reverse();
                for (let j in sortedIndices){
                    const sequence = precedingTokens[i][j].dataSync() + [this.eot];
                    sequences[sequence] = sumLogprobs[i][j].dataSync();
                    if (sequences.length >= this.beamSize){
                        break;
                    }
                }
            }
        }

        let tokens_ = []
        let sumLogprobs_ = []
        for (let sequences in this.finishedSequences){
            for (let seq in Object.keys(sequences)){
                tokens_.push(tf.tensor(seq));
            }
            let values = []
            for (let value in Object.values(sequences)){
                values.push(value)
            }
            sumLogprobs_.push(values)
        }
        return [ tokens_, sumLogprobs_ ]
    }
}
