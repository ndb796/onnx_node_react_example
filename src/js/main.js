// ======================================================================
// Global variables
// ======================================================================

const WIDTH = 64;
const INPUT_SHAPE = [1, 1, WIDTH, WIDTH];
const MAX_LENGTH = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2] * INPUT_SHAPE[3];
const classes = ["Neutral", "Happiness", "Surprise", "Sadness" , "Anger", "Disgust", "Fear", "Contempt"]

let predictedClass;
let predictedConfidence;
let isRunning = false;

// ======================================================================
// DOM Elements
// ======================================================================

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

document.getElementById("file-in").onchange = function (evt) {
  let target = evt.target;
  let files = target.files;

  if (FileReader && files && files.length) {
    isRunning = true;
    var fileReader = new FileReader();
    fileReader.onload = () => onLoadImage(fileReader);
    fileReader.readAsDataURL(files[0]);
  }
};

const target = document.getElementById("target");
window.setInterval(function() {
  if (isRunning) {
    // target.innerHTML = `<img src="src/images/loading.gif" class="loading"/>`;
    target.innerHTML = `<img src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" class="loading"/>`;
  } else if (predictedClass !== undefined) {
    target.innerHTML = `The predicted result is <b>${predictedClass}</b> (${predictedConfidence}%).`;
  } else {
    target.innerHTML = ``;
  }
}, 300);

// ======================================================================
// Functions
// ======================================================================

function onLoadImage(fileReader) {
  var img = document.getElementById("original-image");
  img.onload = () => handleImage(img, WIDTH);
  img.src = fileReader.result;
}

function handleImage(img, targetWidth) {
  ctx.drawImage(img, 0, 0);
  const resizedImageData = resizeImage(img, targetWidth);
  const inputTensor = imageDataToTensor(resizedImageData, INPUT_SHAPE);
  run(inputTensor);
}

function resizeImage(img, width) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = width;
  // canvas.height = canvas.width * (img.height / img.width);
  canvas.height = width;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  document.getElementById("resized-image").src = canvas.toDataURL();
  return ctx.getImageData(0, 0, width, width).data;
}

// Reference: https://github.com/microsoft/onnxruntime-web-demo/blob/main/src/components/models/Emotion.vue
function imageDataToTensor(data, dims) {
  // transpose from [64, 64, 3] -> [1, 64, 64]
  const pixel_array = [];
  for (let i = 0; i < data.length; i += 4) {
    r = data[i];
    g = data[i + 1];
    b = data[i + 2];
    // pixel = (r + g + b) / 3;
    pixel = (r * 0.299 + g * 0.587 + b * 0.114) - 127.5; // normalization for the FER+ model.
    pixel_array.push(pixel);
    // here we skip data[i + 3] because it's the alpha channel (filter out the alpha channel)
  }
  const transposedData = pixel_array;

  // convert to float32
  let l = transposedData.length; // length, we need this for the loop
  const float32Data = new Float32Array(MAX_LENGTH); // create the Float32Array for output
  for (let i = 0; i < l; i++) {
    // float32Data[i] = transposedData[i] / MAX_SIGNED_VALUE; // convert to float
    float32Data[i] = transposedData[i] / 127.5; // convert to float
  }

  // return ort.Tensor
  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}

function argMax(arr) {
  let max = arr[0];
  let maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }
  return [max, maxIndex];
}

async function run(inputTensor) {
  try {
    // load the emotion-ferplus model and create a new inference session.
    // Model File Reference: https://github.com/microsoft/onnxruntime-web-demo/tree/main/public
    const model = await ort.InferenceSession.create('/src/assets/emotion.onnx');

    // console.log("inputNames:", model.inputNames);
    // console.log("outputNames:", model.outputNames);

    // prepare feeds. use model input names as keys.
    const feeds = { Input2505: inputTensor };

    const start = new Date(); // start of an inference.

    // feed inputs and run
    const results = await model.run(feeds);
    const data = results.Softmax2997_Output_0.data;
    // console.log(data);

    // post-process the output data.
    const [maxValue, maxIndex] = argMax(data);
    // console.log(maxValue, maxIndex);
    predictedClass = `${classes[maxIndex]}`;
    predictedConfidence = `${(maxValue * 100).toFixed(4)}`;
    // console.log("predictedClass:", predictedClass);
    // console.log("predictedConfidence:", maxValue);

    const end = new Date(); // end of the inference.
    const inferenceTime = (end.getTime() - start.getTime());
    console.log("inferenceTime:", inferenceTime, "ms.");

    isRunning = false;
  } catch (e) {
    console.error(e);
    isRunning = false;
  }
}