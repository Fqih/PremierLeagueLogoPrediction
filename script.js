let videoElement = document.getElementById('video-preview');
let canvasElement = document.getElementById('canvas');
let canvasContext = canvasElement.getContext('2d');
let stream = null;
let model = null;

async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
}

async function startCameraAndPredict() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = function() {
            setInterval(async function() {
                canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                const imageData = canvasContext.getImageData(0, 0, canvasElement.width, canvasElement.height);
                const tensor = preprocessImage(tf.browser.fromPixels(imageData));
                const prediction = model.predict(tensor);
                const classIndex = tf.argMax(prediction, 1).dataSync()[0];
                const classNames = [
                    'Everton', 'Burnley', 'Newcastle', 'Brighton', 'Wolves',
                    'Tottenham', 'Crystal Palace', 'West Ham', 'Brentford',
                    'Southampton','Arsenal', 'Manchester City', 'Aston Villa',
                    'Liverpool', 'Leicester City', 'Watford', 'Leeds', 'Chelsea',
                    'Manchester United', 'Norwich'
                ];
                document.getElementById('result').innerText = `Prediction: ${classNames[classIndex]}`;
            }, 10);
        };
    } catch (err) {
        console.error('Error accessing the camera: ', err);
    }
}

function preprocessImage(image) {
    const targetSize = [140, 140];
    const resizedImage = tf.image.resizeBilinear(image, targetSize);
    const normalizedImage = resizedImage.div(255.0);
    const batchedImage = normalizedImage.expandDims(0);
    return batchedImage;
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadModel();
    startCameraAndPredict();
});
