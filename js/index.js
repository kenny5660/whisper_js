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
import formatBytes from "format-bytes"
import { Weights } from './Weights';
import { WeightsDownloader } from './WeightsDownloader';
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
        this.mdlProgressInitDone = false;
        let self = this;
        this.weightsDownloader = new WeightsDownloader(this.gui_model_progressbar,this.gui_model_status);
    }
    
    onchangeSelectModel(obj) {
        console.log("Select model :", obj.value);
        this.current_model_name = obj.value;
        this.modelReady = false;
        this.weightsDownloader.abort();
        let self = this;
        this.weightsDownloader.downloadModel(this.current_model_name).then((weights) => this.weightsReady(self,weights));
    }
    weightsReady(self,weights) {
        self.weights = weights
        self.modelReady = true;
        console.log("weightsReady ", this.current_model_name);
        console.log(self.weights)
    }
   
}

var app = new App()