
import React, { useState, useEffect } from 'react';
import {Button, Grid} from '@mui/material'





function Main(props) {
  

    return (
  <div >
    <h1>Graphcore Inference Server</h1>
    <p>This site is a front end for a generic inference service running on IPUs. The goals of this project are to  : 
         <ul>
             <li>Provide a platform to deploy optimized graphcore models.</li>
             <li>Show off graphcore functionality/operation at scale.</li>
             <li>Allow an easy method to show off models with minimal work</li>
         </ul>
         <i>The project is in an alpha state with basic functionality and behavior with room for optimization and improved ease 
         of use.</i>  
    </p>

    <h2>Architecture</h2>
    <img src="base.drawio.png" alt="Diagram of Inference Server Engine"></img>
    <p>The architecture consists of 5 main separable/independent components with generic interfaces to allow other solutions to 
        be substituted. There are numerous inference solutions available but none of them directly support IPUs and surprisingly
        none of them are general. This solution is general and should work for any model with close to optimal performance. </p>
    <ul>
        <li>Model
            <ul>
                <li>Free Running IPU Model with Python and ZMQ Interface</li>
                <li>Support for Streaming and Parallelism Using Multiple Input Queues</li>
            </ul>
        </li>
        <li>Inference Server</li>
            <ul>
                <li><a href="https://developer.nvidia.com/nvidia-triton-inference-server">Triton</a> Based Engine with Python Backend</li>
                <li>PopEF Could Also Be Supported</li>
            </ul>
        <li>API Server <a href="https://fastapi.tiangolo.com/">FastAPI</a>
            <ul>
                <li>User Friend API for Ease of Use</li>
                <li>For NLP Handles Tokenization and Higher Level Processing</li>
            </ul>
        </li>
        <li>Web Client : Javascript/React</li>
        <li>Programmatic Client : Currently Python Only but Easy to Generalize</li>
    </ul>
    <h2>Model Integration</h2>
    <p>The goal of this project is ease of use so both the project configuration and model configuration is a simple 
        process consisting of a simple file addition. The project and model configuration shown below can be set up and run 
        by creating instances of the python classes below.    
        <ul>
            <li>Project Configuration : <a href="https://phabricator.sourcevertex.net/diffusion/PUBLICEXAMPLES/browse/andy%252Fbert_inerference/applications/inference/model_proto/project_proto.py" target="_blank">project_proto.py</a></li>
            <li>Model Configuration : <a href="https://phabricator.sourcevertex.net/diffusion/PUBLICEXAMPLES/browse/andy%252Fbert_inerference/applications/inference/model_proto/model_proto.py" target="_blank">model_proto.py</a></li>
        </ul>
    </p>
    <p>The model itself can be integrated by adding a simple run command to an existing model. Inference has not been a first
        class citizen for the IPU software stack so there might be additional work required on the model side for optimal performance. 
    </p>
  </div>
            
        
       

    );

}

export default Main;
