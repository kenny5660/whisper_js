import fetchProgress from "fetch-progress"
import formatBytes from "format-bytes"
import { Weights } from './Weights';

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
export class WeightsDownloader{
    constructor(gui_model_progressbar,gui_model_status) {
        this.weights = null;
        this.modelReady = false;
        this.current_model_name = ""
        this.gui_model_progressbar = gui_model_progressbar;
        this.gui_model_status = gui_model_status;
        this.mdlProgressInitDone = false;
        this.abortController = new AbortController();
        let self = this;
        this.gui_model_progressbar.addEventListener('mdl-componentupgraded', function () {
            console.log("mdl-componentupgraded");
            this.MaterialProgress.setProgress(0);
            self.mdlProgressInitDone = true;
          });
    }
    abort() {
        console.log("abort");
        this.abortController.abort();
    }
    async downloadModel(modelName) {
        const self = this;
        this.modelReady = false;
        this.current_model_name = modelName;
        this.abortController = new AbortController();
        const status_text = `Try fetch model: "${modelName}" `;
        console.log(status_text);
        self.gui_model_status.innerHTML = status_text;
        let response = await fetch(
            MODELS_URL[this.current_model_name],
            {
                signal: this.abortController.signal
              }
        ).then(
            fetchProgress({
                onProgress: (p)=>this._updateProgressBar(self,p),
            })
            ,
            (err) => this._downloadModelError(self,err)
        )
        console.log("Download done")
        self.weights = new Weights(self.current_model_name)
        let buffer = await response.arrayBuffer()
        await self.weights.init_weights(new Uint8Array(buffer)).then(() => this._weightsDone(self),() => this._weightsError(self))
        return this.weights
    }
    _updateProgressBar(self, progress) {
        const status_text = `Downloading ${self.current_model_name} ${formatBytes(progress.transferred)}/${formatBytes(progress.total)}`;
        const percent = progress.transferred / progress.total  * 100;
        console.log(status_text)
        self.gui_model_status.innerHTML = status_text
        if (self.mdlProgressInitDone) {
            console.log("Update progress bar")
            self.gui_model_progressbar.MaterialProgress.setProgress(percent)
        }
    }

    _downloadModelError(self, err) {
        const status_text = `Model "${self.current_model_name}" download error`;
        if (err instanceof AbortSignal) {
            throw err;
        } else {
            console.log(status_text)
            console.error(err);
            self.gui_model_status.innerHTML = status_text;
            throw err;
        }
        
    }
    _weightsDone(self) {
        const status_text = `Model "${self.current_model_name}" ready!`;
        console.log(status_text);
        self.gui_model_status.innerHTML = status_text;
        self.modelReady = true;
    }
    _weightsError(self) {
        const status_text = `Model "${self.current_model_name}" parse error!`;
        console.log(status_text);
        self.gui_model_status.innerHTML = status_text;
        self.modelReady = false;
    }
}