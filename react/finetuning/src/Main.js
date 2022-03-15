
import React, { useState, useEffect } from 'react';
import {Button, Grid} from '@mui/material'





function Main(props) {
  

    return (
  <div >
    <h1>Graphcore Fine Tuning Server</h1>
    <p>This site is a front end for a fine tuning service running on IPUs. The goals of this project are to  : 
         <ul>
             <li>Provide a platform to deploy optimized graphcore models.</li>
             <li>Show off graphcore functionality/operation at scale.</li>
             <li>Allow an easy method to show off models with minimal work</li>
         </ul>
         <i>The project is in an alpha state with basic functionality and behavior with room for optimization and improved ease 
         of use.</i>  
    </p>

    <h2>Architecture</h2>
    <img src="fine-tuning.png" alt="Diagram of Inference Server Engine"></img>
    <p>The architecture consists of 5 main separable/independent components with generic interfaces to allow other solutions to 
        be substituted. There are numerous inference solutions available but none of them directly support IPUs and surprisingly
        none of them are general. This solution is general and should work for any model with close to optimal performance. </p>
    <ul>
        <li>Model
            <ul>
                <li>Multiple Workers Attatching to Celery (Python) Task Queue</li>
                <li>Model Runs on Request based on Input Configuration</li>
                <li>Supports Scalability through Worker Registration</li>
            </ul>
        </li>
        
        <li>Task Queue <a href="https://docs.celeryproject.org/en/stable/index.html"> (Celery)</a></li>
            <ul>
                <li>General Task Queue to Support Standard Operations (Create/Cancel/Monitor/...)</li>
            </ul>
        <li>API Server <a href="https://fastapi.tiangolo.com/">FastAPI</a>
            <ul>Interface to Task Queue (Might be Bypassable)</ul>
        </li>
        <li>Results Message Queue/Database</li>
        <ul>
            <ul>Results/Status Streamed to Message Queue for Immediate Availability</ul>
            <ul>Database (MongoDB) for long term persistence</ul>
        </ul>
        <li>Client (React Based Webpage)
            <ul>Supports Input Configuration/Run</ul>
            <ul>Supports Results Viewing</ul>
        </li>
    </ul>

  </div>
            
        
       

    );

}

export default Main;
