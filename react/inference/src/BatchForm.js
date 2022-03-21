

import {Button, Grid, Box} from '@mui/material'
import React, { useState, useEffect } from 'react';

import {FormGroup, FormControlLabel} from '@mui/material';
import {SimpleText, SimpleSelect} from './FormBlocks'
import SelectExampleBatch from './SelectExampleBatch';
import { Checkbox } from '@mui/material';

const examples = [
  {
    name:"imdb",
    model_type:"BERT",
    model_size:"Base",
    checkpoint:"textattack/bert-base-uncased-imdb",
    dataset:"imdb,train,text",
    tokenizer:"bert-base-uncased",
    classifier:"Sequence",
    num_labels:2},
  {
    name:"indonlu",
    model_type:"BERT",
    model_size:"Base",
    checkpoint:"ayameRushia/indobert-base-uncased-finetuned-indonlu-smsa",
    dataset:"indonlu:smsa,train,text",
    tokenizer:"ayameRushia/indobert-base-uncased-finetuned-indonlu-smsa",
    classifier:"Sequence",
    num_labels:3
  },
  {
    name:"ner",
    model_type:"BERT",
    model_size:"Base",
    checkpoint:"dslim/bert-base-NER",
    dataset:"wikitext:wikitext-103-v1,test,text",
    tokenizer:"bert-base-uncased",
    classifier:"Token",
    num_labels:9
  },
  {
    name:"mlm",
    model_type:"BERT",
    model_size:"Base",
    checkpoint:"bert-base-uncased",
    dataset:"wikitext:wikitext-103-v1,test,text",
    tokenizer:"bert-base-uncased",
    classifier:"MLM",
    num_labels:48
  }
]

const exampleNames = ['Imdb','Indonlu','NER','MLM']

function FineForm(props) {
  
  const [example, setExample] = useState("Example 1");
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
  }


  const [runId, setRunId] = useState("");
  const [model, setModel] = useState("BERT");
  const [modelSize, setModelSize] = useState("Base");
  const [tokenizer, setTokenizer] = useState("bert-base-uncased");
  const [checkpoint, setCheckpoint] = useState("bert-base-uncased");
  const [dataset, setDataset] = useState("imdb");
  const [textName, setTextName] = useState("");


  const [classifier, setClassifier] = useState("Sequence");
  const [numLabels, setNumLabels] = useState(3);



  const create_query = () => {
    return {
      name:runId,
      model_type:model,
      checkpoint:checkpoint,
      dataset:dataset,
      tokenizer:tokenizer,
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

    fetch("http://192.168.3.114:8101/infer",requestOptions)
      .then(response => response.json())
      .then(r => {
        console.log(r)
      })


  }


    return (

  <Grid container spacing={2} >
    <Grid item xs={12}>
    <h3>Batch Inference</h3>
    </Grid>
    <Grid container padding={1}>
        <SimpleText label="RunId" value={runId} set={setRunId} grid={3}/>
        <SimpleSelect label="Model" value={model} set={setModel} options={["BERT","GPT2"]} grid={3}/>
        <SimpleSelect label="Size" value={modelSize} set={setModelSize} options={["Base","Medium","Large"]} grid={3}/>
    </Grid>
    <Grid container padding={1}>
      <SimpleText label="dataset" value={dataset} set={setDataset} grid={3}/>
      <SimpleText label="tokenizer" value={tokenizer} set={setTokenizer} grid={3}/>
    </Grid>
    <Grid container padding={1}>
      <SimpleText label="checkpoint" value={checkpoint} set={setCheckpoint} grid={3}/>
    </Grid>
    <Grid container padding={1}>
        <SimpleSelect label="classifier" value={classifier} set={setClassifier} options={["Token","Sequence","MLM"]} grid={3}/>
        <SimpleText label="numLabels" value={numLabels} set={setNumLabels} grid={3}/>
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
