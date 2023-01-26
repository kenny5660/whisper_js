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

import * as tf from '@tensorflow/tfjs';
import fetchProgress from "fetch-progress"
import {formatBytes} from "format-bytes"
import { Weights } from './Weights';
import * as ui from 'material-design-lite';
import 'material-icons/iconfont/material-icons.css';

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
class App{

    constructor() {
        this.weights = null;
        this.modelReady = false;
        this.current_model_name = ""
        this.gui_select_model = document.getElementById("select-model");
        this.gui_model_progressbar = document.getElementById("model-progressbar");
        this.gui_model_status = document.getElementById("model-status");
        this.gui_select_model.onchange = () => this.onchangeSelectModel(this.gui_select_model)
        self.mdlProgressInitDone = false;
        this.counter = 0;
        this.gui_model_progressbar.addEventListener('mdl-componentupgraded', function () {
            console.log("componentupgraded");
            this.MaterialProgress.setProgress(0);
            self.mdlProgressInitDone = true;
          });
        this.gui_model_progressbar.nativeElement.MaterialProgress.setProgress(this.this.counter++)
        
    }
    
    onchangeSelectModel(obj) {
        console.log("Select model :", obj.value);
        this.current_model_name = obj.value;
        this.downloadModel();
    }
    updateProgressBar(self,progress) {
        if (self.mdlProgressInitDone) {
            console.log(`Downloading ${self.current_model_name} ${formatBytes(progress.transferred)}/${formatBytes(progress.total)}`)
            console.log(progress)
            
            self.gui_model_status.innerHTML = `Downloading ${self.current_model_name} ${formatBytes(progress.transferred)}/${formatBytes(progress.total)}`
            self.gui_model_progressbar.MaterialProgress.setProgress(progress.total / progress.transferred * 100)
        }
    }
    downloadModelDone(self,response) {
        console.log("Download done")
        console.log(response)
        console.log(self)
        self.weights = new Weights(self.current_model_name)
        response.arrayBuffer().then((buffer) => self.weights.init_weights(new Uint8Array(buffer)).
            then(() => this.weightsDone(self),() => this.weightsError(self)))
    }

    downloadModelError(self, err) {
        const status_text = `Model "${self.current_model_name}" download error`;
        console.log(status_text)
        console.error(err);
        self.gui_model_status.innerHTML = status_text;
        throw err;
    }
    downloadModel() {
        const self = this;
        this.modelReady = false;
        const status_text = `Try fetch model: "${self.current_model_name}" `;
        console.log(status_text);
        self.gui_model_status.innerHTML = status_text;
        fetch(MODELS_URL[this.current_model_name]).then(
            fetchProgress({
                onProgress: (p)=>this.updateProgressBar(self,p),
                onError: (err) => this.downloadModelError(self,err)
            })
            ,
            (err) => this.downloadModelError(self,err)
        ).then((resp)=>this.downloadModelDone(self,resp))
    }
    weightsDone(self) {
        const status_text = `Model "${self.current_model_name}" ready!`;
        console.log(status_text);
        self.gui_model_status.innerHTML = status_text;
        self.modelReady = true;
    }
    weightsError(self) {
        const status_text = `Model "${self.current_model_name}" parse error!`;
        console.log(status_text);
        self.gui_model_status.innerHTML = status_text;
        self.modelReady = false;
    }
}

var app = new App()