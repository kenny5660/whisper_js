const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

import * as tf from "@tensorflow/tfjs";
import filters from './filters.json';

function exact_div(x, y){

    return Math.floor(x/y);

};

function log10(x){

    let x1 = tf.log(x);
    let x2 = tf.log(10.0);

    return tf.div(x1, x2);
};

// hard-coded audio hyperparameters
const SAMPLE_RATE = 16000
const N_FFT = 400
const N_MELS = 80
const HOP_LENGTH = 160
const CHUNK_LENGTH = 30
const N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  // 480000: number of samples in a chunk
const N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  // 3000: number of frames in a mel spectrogram input


export async function loadAudio(arrayBuffer) {

    let data = await audioCtx.decodeAudioData(arrayBuffer);
    console.log(arrayBuffer);

    let offlineCtx = new OfflineAudioContext(data.numberOfChannels,
                                             data.duration * SAMPLE_RATE,
                                             SAMPLE_RATE);
    
    let offlineSource = offlineCtx.createBufferSource();
    offlineSource.buffer = data;
    offlineSource.connect(offlineCtx.destination);
    offlineSource.start();
    let resampled = await offlineCtx.startRendering();

    let decodedData = new Float32Array(resampled.length);

    resampled.copyFromChannel(decodedData, 0, 0);

    return tf.tensor(decodedData);

};

export async function logMelSpectrogram(fileDirectory) {

    let stft = tf.signal.stft(audio, N_FFT, HOP_LENGTH, N_FFT, tf.signal.hannWindow);
    let magnitudes = tf.abs(stft).pow(2).transpose();
    let melSpec = tf.matMul(filters, magnitudes);

    let logSpec = tf.clipByValue(melSpec, 1e-10, melSpec.max().arraySync());
    logSpec = log10(logSpec);
    logSpec = tf.maximum(logSpec, tf.sub(logSpec.max(), 8.0).arraySync());
    logSpec = tf.div(tf.add(logSpec, 4.0), 4.0);

    if (logSpec.shape[1] < N_FRAMES){
        logSpec = tf.pad(logSpec, [[0, 0], [0, N_FRAMES - logSpec.shape[1]]]);
    };
    if(logSpec.shape[1] > N_FRAMES){
        logSpec = logSpec.slice([0, 0], [N_MELS, 3000])
    }

    return logSpec;

};