

import {Button, Grid, Box} from '@mui/material'
import React, { useState, useEffect } from 'react';

import {FormControl, FormControlLabel, FormLabel, RadioGroup, Radio, TextField, Select, MenuItem, InputLabel} from '@mui/material';
import {SimpleText, SimpleSelect} from './FormBlocks'
import SelectExampleBatch from './SelectExampleBatch';


function FineForm(props) {
  
  const examples = [
    {
      name:"imdb",
      model_type:"BERT",
      model_size:"Base",
      checkpoint:"bert-base-uncased",
      dataset:"imdb",
      tokenizer:"bert-base-uncased",
      classifier:"Sequence",
      num_labels:2, 
      cloud:"None",
      endpoint:"",
      results:""
    },
    {
      name:"imdb_cloud",
      model_type:"BERT",
      model_size:"Base",
      checkpoint:"bert-base-uncased",
      dataset:"imdb",
      tokenizer:"bert-base-uncased",
      classifier:"Sequence",
      num_labels:2, 
      cloud:"AzureBlob",
      endpoint:"DefaultEndpointsProtocol=https;AccountName=andynlpstore;AccountKey=hkMiWLiqIpONH0NnyhmYAO9SmdVJZb1CazjCB6mnk/72ee5KdyKnq/ByS5s6/ZPUPbP2HImIveIvwxYSP88Reg==;EndpointSuffix=core.windows.net",
      results:""
    }
  ]

  const exampleNames = ["Imdb","Imdb(Cloud)"]
  const updateExample = (x) => {
    console.log(examples[x].model_type)
    setRunId(examples[x].name);
    setModel(examples[x].model_type);
    setModelSize(examples[x].model_size);
    setTokenizer(examples[x].tokenizer);
    setCheckpoint(examples[x].checkpoint);
    setDataset(examples[x].dataset);
    setClassifier(examples[x].classifier);
    setNumLabels(examples[x].num_labels);
    setCloud(examples[x].cloud);
    setEndPoint(examples[x].endpoint);
    setResultsFolder(examples[x].results)
  }


  const [runId, setRunId] = useState("");
  const [model, setModel] = useState("BERT");
  const [modelSize, setModelSize] = useState("Base");
  const [tokenizer, setTokenizer] = useState("bert-base-uncased");
  const [checkpoint, setCheckpoint] = useState("bert-base-uncased");
  const [dataset, setDataset] = useState("imdb");
  const [optimizer, setOptimizer] = useState("ADAM");
  const [learningRate, setLearningRate] = useState(.0001);
  const [epochs, setEpochs] = useState(1);

  const [classifier, setClassifier] = useState("Sequence");
  const [numLabels, setNumLabels] = useState(2);
  const [endpoint, setEndPoint] = useState("");
  const [cloud, setCloud] = useState("None");
  const [resultsFolder, setResultsFolder] = useState("");


  const create_query = () => {
    return {
      name:runId,
      model_type:model,
      checkpoint:checkpoint,
      dataset:dataset,
      tokenizer:tokenizer,
      optimizer:optimizer,
      learning_rate:learningRate,
      epochs:epochs,
      classifier:classifier,
      num_labels:numLabels

    }
    
  }

  const submit = (x) => {
    const formData = create_query()
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    };
    //console.log(requestOptions)
    var time = new Date().getTime()

    fetch("http://192.168.3.114:8101/tune",requestOptions)
      .then(response => response.json())
      .then(r => {
        console.log(r)
      })


  }


    return (

  <Grid container spacing={2} >
    <Grid item xs={12}>
    <h3>Fine Tuning</h3>
    </Grid>
    <Grid container padding={1}>
        <SimpleText label="RunId" value={runId} set={setRunId} grid={3}/>
        <SimpleSelect label="Model" value={model} set={setModel} options={["BERT","GPT2"]} grid={3}/>
        <SimpleSelect label="Size" value={modelSize} set={setModelSize} options={["Base","Medium","Large"]} grid={3}/>
    </Grid>
    <Grid container padding={1}>
        <SimpleSelect label="classifier" value={classifier} set={setClassifier} options={["Token","Sequence","MLM"]} grid={3}/>
        <SimpleText label="numLabels" value={numLabels} set={setNumLabels} grid={3}/>
    </Grid>
    <Grid container padding={1}>
      <SimpleSelect label="cloud" value={cloud} set={setCloud} options={["None","AzureBlob"]} grid={3}/>
      <SimpleText label="cloud_endpont" value={endpoint} set={setEndPoint} grid={6}/>
    </Grid>

    <Grid container padding={1}>
        <SimpleText label="tokenizer" value={tokenizer} set={setTokenizer} grid={3}/>
        <SimpleText label="checkpoint" value={checkpoint} set={setCheckpoint} grid={3}/>
        <SimpleText label="dataset" value={dataset} set={setDataset} grid={3}/>
    </Grid>

    <Grid container padding={1}>
        <SimpleSelect label="optimizer" value={optimizer} set={setOptimizer} options={["ADAM","LAMB"]} grid={3}/>
        <SimpleText label="learning-rate" value={learningRate} set={setLearningRate} grid={3}/>
        <SimpleText label="epochs" value={epochs} set={setEpochs} grid={3}/>
    </Grid>
    <Grid container padding={1}>
      <Grid item xs={2}>
         <Button variant="contained" color="primary" type="submit" size="small" onClick={submit}>Submit</Button>
      </Grid>
      <Grid item xs={2} ><SelectExampleBatch exampleNames={exampleNames} updateExample={updateExample}/></Grid>
    </Grid>
  
  </Grid>


  
)

}

export default FineForm;