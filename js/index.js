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
import * as loader from './weights_loader';
import * as ui from 'material-design-lite';
import 'material-icons/iconfont/material-icons.css';

class App{

    constructor() {
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
        this.downloadModel(obj.value);
    }
    updateProgressBar() {
        if (self.mdlProgressInitDone) {
            console.log("update");
            this.gui_model_progressbar.MaterialProgress.setProgress(this.counter++)
        }
    }
    downloadModel(model) {
        console.log("Download model :", model);
        window.setInterval(() => this.updateProgress(), 200);
        
    }
}

var app = new App()