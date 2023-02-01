const tf = require('@tensorflow/tfjs');

/*
Выбирает сэмплы с наибольшой вероятностью.
Вероятности нормализуются на длину токенов или по правилу Google NMT.
from the Google NMT paper: penalty = ((5 + length) / 6) ** self.length_penalty
*/

class MaximumLikelihoodRanker {
    constructor(lengthPenalty) {
        this.lengthPenalty = lengthPenalty;
    }

     rank(tokens, sumLogProbs) {
        let result = []
        const scores = this.getScores(tokens, sumLogProbs);
        for (let i = 0; i < scores.length; i++){
            result.push(tf.argMax(scores[i]))
        }
        return result
    }

     getScores(tokens, sumLogProbs) {
        let result = [];

        let penalties = tf.tensor1d(tokens.map(x => x.length));
        if (this.lengthPenalty) {
            penalties = penalties.mul(this.lengthPenalty).pow((penalties.add(5)).div(6));
        }

        penalties = penalties.dataSync();

        for (let i = 0; i < sumLogProbs.length; i++) {
            result.push(sumLogProbs[i].div(penalties[i]))
        }
        return result
    }
}

