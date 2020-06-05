const imageUpload = document.getElementById('imageUpload');

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('node_modules/face-api.js/dist/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('node_modules/face-api.js/dist/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('node_modules/face-api.js/dist/models')
]).then(initiate)

 async function initiate(){
    const container = document.createElement('div');
    container.style.position = 'relative';
    document.body.append(container);
    const LabeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(LabeledFaceDescriptors, 0.6)
    document.body.append('loaded');
    let image
    let canvas

    imageUpload.addEventListener('change', async () => {
        if(image) image.remove()
        if(canvas) canvas.remove()
        const image = await faceapi.bufferToImage(imageUpload.files[0]);
        container.append(image);
        const canvas = faceapi.createCanvasFromMedia(image);
        container.append(canvas);

        const displaySize = {width: image.width, height: image.height}
        faceapi.matchDimensions(canvas, displaySize)
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        const results = resizedDetections.map(detect => faceMatcher.findBestMatch(detect.descriptor))
        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString()})
            drawBox.draw(canvas)
        })
    })
}

function loadLabeledImages(){
    const labels = ['Firmino', 'Mane', 'Trent']
    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for(let i = 1; i <=2; i++){
                const img = await faceapi.fetchImage(`https://drive.google.com/drive/folders/1Xq913_COUQ94tXcXIqU-NnOkwW_OFnVN?usp=sharing
                /${label}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}