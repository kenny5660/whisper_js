# Whisper js

Whisper_js is port of [Whisper speech recognition model](https://github.com/openai/whisper) to JavaScript.  
It allows to run whisper locally in the browser, without server side. But client should have a powerful videocard.
Port using [TensorFlow.js](https://github.com/tensorflow/tfjs) as machine learning library.  

`Beam num` input is config for beamSerachDecoder, 0 - greedy decoder.
## Live DEMO

[GitHubPages demo](https://kenny5660.github.io/whisper_js/)

## Issues
- Memory leek after each recognition
- Duration of the audio file must be less than 30 seconds
- Beam num greater than 0 works unstable.
- Multilanguage dont work
## Launch DEMO localy

To launch the demo localy, do

```sh
yarn
yarn watch
```
