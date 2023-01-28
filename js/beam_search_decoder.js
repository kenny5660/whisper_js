tf = require('@tensorflow/tfjs');

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

    update(tokens, logits) {
        if (tokens.shape[0] % this.beamSize !== 0) {
            throw new Error(`${tokens.shape[0]} % ${this.beamSize} !== 0`);
        }

        const nAudio = tokens.shape[0] / this.beamSize;
        if (this.finishedSequences === null) {
            this.finishedSequences = Array(nAudio).fill({});
        }

        const logProbs = tf.softmax(logits).log();
        const nextTokens = [];
        const sourceIndices = [];
        const finishedSequences = Array(nAudio).fill({});

        for (let i = 0; i < nAudio; i++) {
            const scores = {};
            const sources = {};
            const finished = {};

            for (let j = 0; j < this.beamSize; j++) {
                const idx = i * this.beamSize + j;
                const prefix = tokens.slice([idx], [1]).arraySync()[0];

                for (let k = 0; k < this.beamSize + 1; k++) {
                    const value = logProbs.slice([idx, k], [1, 1]).arraySync()[0][0];
                    const token = k;
                    const sequence = [...prefix, token];
                    scores[sequence] = value;
                    sources[sequence] = idx;
                }
            }

            let saved = 0;
            for (const sequence of Object.keys(scores).sort((a, b) => scores[b] - scores[a])) {
                if (sequence[sequence.length - 1] === this.eot) {
                    finished[sequence] = scores[sequence];
                } else if (saved < this.beamSize) {
                    nextTokens.push(sequence);
                    sourceIndices.push(sources[sequence]);
                    saved++;
                }
            }

            finishedSequences[i] = finished;
        }

        tokens = tf.tensor(nextTokens);
        this.inference.rearrangeKvCache(sourceIndices);

        for (let i = 0; i < this.finishedSequences.length; i++) {
            for (const seq of Object.keys(finishedSequences[i]).sort((a, b) => finishedSequences[i][b] - finishedSequences[i][a])) {
                if (Object.keys(this.finishedSequences[i]).length >= this.maxCandidates) {
                    break;
                }
                this.finishedSequences[i][seq] = finishedSequences[i][seq];
            }
        }

        const completed = this.finishedSequences.every(sequences => Object.keys(sequences).length >= this.maxCandidates);
        return {tokens, completed};
    }

    finalize(precedingTokens, sumLogprobs) {
        let results = Array(this.finishedSequences.length).fill([]);
        for (let i = 0; i < this.finishedSequences.length; i++) {
            for (const seq of Object.keys(this.finishedSequences[i]).sort((a, b) => this.finishedSequences[i][b] - this.finishedSequences[i][a])) {
                results[i].push(seq);
                if (results[i].length >= this.beamSize) {
                    break;
                }
            }
            if (results[i].length < this.beamSize) {
                for (let j = 0; j < sumLogprobs.shape[1]; j++) {
                    if (precedingTokens.slice([i, j], [1, 1]).arraySync()[0][0] !== this.eot) {
                        results[i].push(precedingTokens.slice([i, j], [1, 1]).arraySync()[0]);
                        if (results[i].length >= this.beamSize) {
                            break;
                        }
                    }
                }
            }
        }
        return results;
    }

}
