/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


import { WeightsDownloader } from './WeightsDownloader';
import * as ui from 'material-design-lite';
import 'material-icons/iconfont/material-icons.css';
import { Whisper } from './whisper/model.js';
import * as tf from '@tensorflow/tfjs';
import * as load_audio from "./load_audio.js"
const MODELS_URL = {
    "tiny.en": "tiny.en.h5",
    "tiny": "tiny.h5",
    "base.en": "base.en.h5",
    "base": "base.h5",
    "small.en": "small.en.h5",
    "small": "small.h5",
    "medium.en": "medium.en.h5",
    "medium": "medium.h5",
    "large": "large.h5"
}

function preprocessAudio(file) {
    return file.arrayBuffer()
        .then(load_audio.loadAudio)
        .then(load_audio.logMelSpectrogram);
};

class App {

    constructor() {
        this.weights = null;

        this.modelReady = false;
        this.current_model_name = ""
        this.gui_model_output = document.getElementById("model-output");
        this.gui_select_model = document.getElementById("select-model");
        this.gui_model_progressbar = document.getElementById("model-progressbar");

        this.gui_model_status = document.getElementById("model-status");
        this.gui_select_model.onchange = () => this.onchangeSelectModel(this.gui_select_model)

        this.input_audiofile = document.getElementById("input_audiofile");
        this.input_audiofile.onchange = (event) => this.onchangeInputAudioFile(event);

        this.record_audio = document.getElementById("input-start-stream");
        this.record_audio.onclick = (event) => this.onclickRecordAudioFile();
        

        this.mdlProgressInitDone = false;
        let self = this;
        this.weightsDownloader = new WeightsDownloader(this.gui_model_progressbar, this.gui_model_status);
        this.whisper = null

        this.recording_audio = false;
    }
    async onchangeInputAudioFile(event) {
        let file = this.input_audiofile.files[0]
        console.log(file);
        if (file.name.split('.').pop() == "json") {
            file.text().then(JSON.parse).then(res => tf.tensor(res.mel)).then((res) => this.run_model(res)).then(res => console.log(res));
        }
        else {
            const audio = await preprocessAudio(file);
            this.run_model(audio).then( (res) =>this.print_model_result(res));
        }
    }

    async onclickRecordAudioFile() {
        if (!this.recording_audio) {
            this.record_audio.children[0].innerHTML = 'mic_on';

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.rec = new MediaRecorder(stream);

            let chunks = [];
            this.rec.ondataavailable = e => chunks.push(e.data);
            this.rec.onstop = async e => {
                const blob = new Blob(chunks, { type: "audio/ogg; codecs=opus" });
                const url = window.URL.createObjectURL(blob);

                const file = await fetch(url);
                const audio = await preprocessAudio(file);

                audio.print();
                this.run_model(audio).then( (res) =>this.print_model_result(res))

            }
            this.rec.start();
            this.recording_audio = true;
        } else {
            this.record_audio.children[0].innerHTML = 'mic_off';
            this.rec.stop();
            this.recording_audio = false;
        }
    }

    onchangeSelectModel(obj) {
        console.log("Select model :", obj.value);
        this.current_model_name = obj.value;
        this.modelReady = false;
        this.weightsDownloader.abort();
        let self = this;
        this.weightsDownloader.downloadModel(this.current_model_name).then((weights) => this.weightsReady(self, weights));
    }
    weightsReady(self, weights) {
        self.weights = weights
        self.whisper = new Whisper(self.weights.weights);
        console.log("weightsReady ", this.current_model_name);
        console.log(self.weights)
        self.modelReady = true;
    }
    async run_model(mel_audio) {
        console.time("run_model");
        console.timeLog("run_model", "Run whisper");
        let result = this.whisper.decode(mel_audio);
        console.timeLog("run_model", "Recognition completed");
        console.timeEnd("run_model");
        return result;
    }
    async print_model_result(decoding_result) {
        console.log(decoding_result)
        this.gui_model_output.innerText = decoding_result.text;
    }

}

navigator.permissions.query({ name: 'microphone' }).then(function (result) {
    if (result.state == 'granted') {
        console.log('success');
    } else if (result.state == 'prompt') {
    } else if (result.state == 'denied') {
        alert('Try again');
    };
    result.onchange = function () { };
});


var app = new App()