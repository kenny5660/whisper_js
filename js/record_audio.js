navigator.permissions.query({name: 'microphone'}).then(function (result) {
    if (result.state == 'granted') {
    } else if (result.state == 'prompt') {
    } else if (result.state == 'denied') {
    }
    result.onchange = function () {};
  });

navigator.mediaDevices.getUserMedia({audio:true})
.then(stream => {handlerFunction(stream)})


function handlerFunction(stream) {
    rec = new MediaRecorder(stream);
    rec.ondataavailable = e => {
        audioChunks.push(e.data);
        if (rec.state == "inactive"){
        let blob = new Blob(audioChunks,{type:'audio/mpeg-3'});
        return blob.arrayBuffer(); 
        }
    }
}

record.onclick = e => {
    console.log('Record was clicked')
    record.disabled = true;
    stopRecord.disabled=false;
    audioChunks = [];
    rec.start();
}
stopRecord.onclick = e => {
    console.log("Stop was clicked")
    record.disabled = false;
    stop.disabled=true;
    rec.stop();
}