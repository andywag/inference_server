import TextField from '@mui/material/TextField';
import {Button, Grid} from '@mui/material'
//import SquadTable from './SquadTable'
import React, { useState, useEffect } from 'react';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem'
import OutlinedInput from '@mui/material/OutlinedInput';
import { useTheme } from '@mui/material/styles';

import SelectExample from './SelectExample';
import ReactJson from 'react-json-view'
import AppBar from '@mui/material/AppBar';
import {ONLINE_ADDRESS} from './Constants'

/*
const requestOptions = {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: "", answer:"" })
};
*/



function Squad(props) {
  
  const examples = [
    {'question':'Where do I live?','context':'My name is Wolfgang and I live in Berlin'},
    {'question':"What's my name?",'context':'My name is Clara and I live in Berkeley.'},
    {'question':'Which name is also used to describe the Amazon rainforest in English?',
    'context':'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America'}

  ]


  const [question, setQuestion] = useState(examples[0].question);
  const [answer, setAnswer] = useState(examples[0].context);
  const [result, setResult] = useState("");
  const [jsonResponse, setJsonResponse] = useState({})
  const [responseTime, setResponseTime] = useState(0)
  const [serverTime, setServerTime] = useState(0)

  const [example, setExample] = useState("Example 1");
  const theme = useTheme();


  
  const ITEM_HEIGHT = 48;
  const ITEM_PADDING_TOP = 8;
  const MenuProps = {
    PaperProps: {
      style: {
        maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
        width: 250,
      },
    },
  };


  const names = [0,1,2]




  const updateExample = (x) => {
    setQuestion(examples[x].question)
    setAnswer(examples[x].context)
  }

  const updateAnswer = (r) => {
    setResult(answer.substring(r.results[0].start,r.results[0].end))
    setJsonResponse(r)
  }

  const query = () => {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, answer:answer})
    };
    //console.log(requestOptions)
    var time = new Date().getTime()

    fetch(ONLINE_ADDRESS + "squad_rabbit",requestOptions)
      .then(response => response.json())
      .then(r => {
 
        setResponseTime(new Date().getTime()-time)
        setServerTime(parseInt(1000*r.server_time))
        updateAnswer(r)
      })
    
    //.then(data => this.setState({ postId: data.id }));
  }

    return (
  <Grid container spacing={2}>
    <Grid item xs={12}>
    <h1>BERT Large Squad Question and Answer (Currently Not Running)</h1>
    <p>Due to IPU availability this model has been disabled.</p>
    </Grid>
    
    <Grid item xs={12}>
    <TextField value={question} label="question" style = {{width: 800}} multiline="True" onChange={(e) => {setQuestion(e.target.value);}} InputLabelProps={{ shrink: true }}/>

    </Grid>
    <Grid item xs={12}>
    <TextField value={answer} label="context" style = {{width: 800}} multiline="True" onChange={(e) => {setAnswer(e.target.value);}} InputLabelProps={{ shrink: true }}/>
    </Grid>

    <Grid item xs={1}>
    <Button variant="contained" onClick={query}>Submit</Button>
    </Grid>
    <Grid item xs={2} ><SelectExample updateExample={updateExample}/></Grid>

    <Grid item xs={2}>
      <p><b>Response Time</b> {responseTime} ms</p>
    </Grid>
    <Grid item xs={2}>
      <p><b>Server Time</b> {serverTime} ms</p>
    </Grid>
    <Grid item xs={12}>
      <TextField value={result} label="response" style = {{width: 800}}/>
    </Grid>
    <Grid item xs={12}>
      <b>API Documentation/Root : <a href="http://192.168.3.114:8100/docs" target="_blank">http://192.168.3.114:8100/docs</a></b>
    </Grid>
    <Grid item xs={12}>
      <b>Model Configuration Source <a href="https://phabricator.sourcevertex.net/diffusion/PUBLICEXAMPLES/browse/andy%252Fbert_inerference/applications/inference/project/squad_proto.py" target="_blank">squad_proto.py</a></b>
    </Grid>
    <Grid item xs={12}>
      <ReactJson collapsed="True" src={jsonResponse}/>
    </Grid>
    
  </Grid>
            
        
       

    );

}

export default Squad;
