import * as tf from '@tensorflow/tfjs';

/*
Выбирает сэмплы с наибольшой вероятностью.
Вероятности нормализуются на длину токенов или по правилу Google NMT.
from the Google NMT paper: penalty = ((5 + length) / 6) ** self.length_penalty
*/


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

//
// const ranker = new MaximumLikelihoodRanker(0.6);
//
// const tokens = [
//     [tf.tensor1d([1, 2, 3]), tf.tensor1d([4, 5, 6]), tf.tensor1d([7, 8, 9])],
//     [tf.tensor1d([10, 11, 12]), tf.tensor1d([13, 14, 15]), tf.tensor1d([16, 17, 18])]
// ];
// const sumLogProbs = [tf.tensor1d([-3.4, -2.5, -1.6]), tf.tensor1d([-4.5, -3.6, -2.7])];
//
// const indices = ranker.rank(tokens, sumLogProbs);
// console.log(indices.dataSync());

