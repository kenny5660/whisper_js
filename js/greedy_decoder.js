const tf = require('@tensorflow/tfjs')

function CTCGreedyDecoder(logits, blankLabel = 0, mergeRepeated = true) {
    let maxScores = tf.argMax(logits, 2).transpose()
    let tensor = maxScores.arraySync();
    let decodes = [];
    for (let batch of tensor) {
        let decode = [];
        for (let j = 0; j < batch.length; j++) {
            let index = batch[j];
            if (index !== blankLabel) {
                if (mergeRepeated && j !== 0 && index === batch[j - 1]) {
                    continue;
                }
                decode.push(index);
            }
        }
        decodes.push(decode);
    }
    return decodes;
}

let T = 2;
let C = 5;
let N = 6;

let logits = tf.randomUniform([T, N, C]);
console.log(logits.arraySync())
let output = CTCGreedyDecoder(logits);
console.log(output);
