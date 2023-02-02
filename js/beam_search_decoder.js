import * as tf from "@tensorflow/tfjs-core";

tf = require('@tensorflow/tfjs');

/*
constructor:
- beamSize количество разных поисков, == 1 почти равно GreedyDecoder
- eot - символ конца последовательности
- inference - класс для кэширования
- patience - коэффициент уверенности
- maxCandidates - максимальное количество гипотез
- finishedSequences - вспомогательное поле для проверки окончания
 */

function sortDictByValue(dict) {
    const sorted = {};
    Object.entries(dict)
        .sort((a, b) => b[1] - a[1])
        .forEach(([key, value]) => {
            sorted[key] = value;
        });
    return sorted;
}


export class BeamSearchDecoder {
    constructor(beamSize, eot, inference, patience = 1.0) {
        this.beamSize = beamSize;
        this.eot = eot;
        this.inference = inference;
        this.patience = patience;
        this.maxCandidates = Math.floor(beamSize * this.patience);
        this.finishedSequences = null;

        if (this.maxCandidates <= 0) {
            throw new Error(`Invalid beam size (${beamSize}) or patience (${patience})`);
        }
    }

    reset() {
        this.finishedSequences = null;
    }

    update(tokens, logits, sumLogprobs) {
        tokens.print();
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
        let completed = false;

        for (let i = 0; i < nAudio; i++) {
            let scores = {};
            const sources = {};
            const finished = {};

            for (let j = 0; j < this.beamSize; j++) {
                const idx = i * this.beamSize + j;
                const prefix = tokens.gather(idx);
                const logProbsIdx = logProbs.gather(idx);
                const {values, indices} = logProbsIdx.topk(this.beamSize + 1);

                for (let k = 0; k < indices.shape[0]; k++) {
                    let newLogprob = (sumLogprobs.gather(idx).add(values.gather(idx))).dataSync();
                    let sequence = prefix.concat(indices.gather(idx).expandDims(0)).dataSync();
                    scores[sequence] = newLogprob;
                    sources[sequence] = idx;
                }
            }


            let saved = 0;
            scores = sortDictByValue(scores)
            for (const sequence in scores) {
                const sequence_int = sequence.split(',').map(Number)
                if (sequence_int[sequence_int.length - 1] === this.eot) {
                    finished[sequence] = scores[sequence];
                    completed = true;
                } else {
                    sumLogprobs = sumLogprobs.dataSync();
                    sumLogprobs[nextTokens.length] = scores[sequence];
                    sumLogprobs = tf.tensor(sumLogprobs);
                    nextTokens.push(sequence);
                    sourceIndices.push(sources[sequence]);
                    saved +=1;
                    if (saved === this.beamSize) {
                        break;
                    }
                }
            }
            finishedSequences.push(finished);
        }
        if (!completed) {
            tokens = nextTokens.map(str => str.split(',').map(num => parseInt(num, 10)));
            tokens = tf.tensor(tokens);
        }
        this.inference.rearrangeKVCache(sourceIndices);

        if (this.finishedSequences.length !== finishedSequences.length) {
            throw new Error(`this.finishedSequences len != finishedSequences.length`);
        }
        // throw new Error()
        for (const key in this.finishedSequences) {
            const previouslyFinished = this.finishedSequences[key];
            let newlyFinished = finishedSequences[key];
            newlyFinished = sortDictByValue(newlyFinished)
            for (const seq in newlyFinished) {
                    if (Object.keys(previouslyFinished).length >= this.maxCandidates) {
                        return;
                    }
                    previouslyFinished[seq] = newlyFinished[seq];
                }
        }
        // const completed = this.finishedSequences.every(sequences => Object.keys(sequences).length >= this.maxCandidates);

        tokens.print();
        sumLogprobs.print();
        return [ tokens, completed ];
    }

    finalize(precedingTokens, sumLogprobs) {
        for (let i = 0; i < this.finishedSequences.length; i++) {
            let sequences = this.finishedSequences[i];
            if (sequences.length < this.beamSize){
                const sortedIndices = Array.from(sumLogprobs.gather(i).argMin().dataSync()).reverse();
                for (let j in sortedIndices){
                    const sequence = precedingTokens.gather(i).gather(j).dataSync() + [this.eot];
                    sequences[sequence] = sumLogprobs.gather(i).gather(j).dataSync();
                    if (sequences.length >= this.beamSize){
                        break;
                    }
                }
            }
        }

        let tokens = [];
        let logprobs = [];

        for (const dict in Object.keys(this.finishedSequences)){
            const sequences = this.finishedSequences[dict]
            for (let seq in sequences){
                logprobs.push(sequences[seq]);
                seq = tf.tensor(seq.split(',').map(num => parseInt(num, 10)));
                tokens.push([seq]);
            }


        }
        console.log(tokens);
        console.log(logprobs);
        return [tf.tensor(tokens), tf.tensor(logprobs)]
    }
}
