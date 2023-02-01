import * as tf from '@tensorflow/tfjs';

export class MaximumLikelihoodRanker {
    constructor(lengthPenalty) {
        this.lengthPenalty = lengthPenalty;
    }

     rank(tokens, sumLogProbs) {
        let result = []
        const scores = this.getScores(tokens, sumLogProbs);
        for (let i = 0; i < scores.length; i++){
            result.push(tf.argMax(scores[i]).arraySync());
        }
        return result
    }

     getScores(tokens, sumLogProbs) {
        let result = [];

        let penalties = tf.tensor(tokens.map(x => x.map(y => y.length)));
        if (this.lengthPenalty) {
            // disabled
            penalties = penalties.mul(this.lengthPenalty).pow((penalties.add(5)).div(6));
        }

        penalties = penalties.dataSync();

        for (let i = 0; i < sumLogProbs.length; i++) {
            result.push([sumLogProbs[i] / penalties[i]]);
        }
        return result
    }
}

