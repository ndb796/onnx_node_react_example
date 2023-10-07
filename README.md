### <b>Node.js Installation</b>

* To use the light-server, we need the *Node.js* runtime.
* <b>Download</b>: https://nodejs.org/en/download
* After the installation, we can use the *npm* and *npx* commands.

### <b>How to Install</b>

<pre>
npm init
</pre>

* After install a <b>Node.js</b> package, the followings will appear:
  1. *package.json* shows the packages we want to use.
  2. *package-lock.json* shows the detailed packages with version names.
  3. *node_modules* contains the whole source codes of installed packages.

<pre>
npm install light-server
</pre>

### <b>How to Run</b>

* After running the light-server, we can open the link *http://localhost:8080/*.

<pre>
npx light-server -s . -p 8080
</pre>

### <b>(Tutorial) How to Use the ONNX Model</b>

* We can simply use the extracted <b>ONNX</b> model file following the below code template.
* We need the "ort.min.js" from the ONNX runtime web library.
  * Option 1. Loading from the ONNX CDN.
  * Option 2. Using *const ort = require('onnxruntime-web');* when using the Webpack.

<pre>
async function run() {
  try {
    // load the emotion-ferplus model and create a new inference session.
    const model = await ort.InferenceSession.create('./emotion-ferplus-7.onnx');

    // define an input shape.
    const inputShape = [1, 1, 64, 64];
    const size = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3];
    // generate a dummy input data.
    const inputData = Float32Array.from({ length: size }, () => Math.random());

    console.log("inputNames:", model.inputNames);
    console.log("outputNames:", model.outputNames);

    // prepare feeds. use model input names as keys.
    const feeds = { Input3: new ort.Tensor('float32', inputData, dims) };

    const start = new Date(); // start of an inference.

    // feed inputs and run
    const results = await model.run(feeds);
    console.log(results.Plus692_Output_0.data);

    const end = new Date(); // end of the inference.
    const inferenceTime = (end.getTime() - start.getTime());
    console.log("inferenceTime:", inferenceTime);
  } catch (e) {
    console.log(e);
  }
}

run();
</pre>

### <b>(Tutorial) How to Use the Webpack</b>

* To use the ONNX web assembly extensions, we can use the plugin method.
* We can write the *webpack.config.js* code.
* After the *npx webpack*, we can access the final bundled main code *bundle.min.js*.

<pre>
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin"); // for using the ONNX extensions.

module.exports = () => {
    return {
        target: ['web'],
        entry: path.resolve(__dirname, 'src/js/main.js'),
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'bundle.min.js',
            library: {
                type: 'umd'
            }
        },
        plugins: [new CopyPlugin({
            // for using the ONNX runtime library, copy *.wasm to the output folder.
            patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
        })],
        mode: 'production'
    }
};
</pre>

### <b>(Tutorial) Git Initialization Using Codes</b>

* Set the Git configuration of my local computer.

<pre>
git config --global user.name "ndb7967"
git config --global user.email "ndb7967@gmail.com"
</pre>

* Initialize the Git project in this directory.

<pre>
git init
</pre>

* Add the files to commit.

<pre>
git add .
git status
</pre>

* Commit the files.

<pre>
git commit -m "Update"
</pre>

* Add the remote GitHub repository.

<pre>
git branch -M main
git remote add origin https://github.com/ndb7967/onnx_example.git
</pre>

* Push the codes.

<pre>
git push -u origin main
</pre>
