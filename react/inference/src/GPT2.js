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

/*
const requestOptions = {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: "", answer:"" })
};
*/



function Squad(props) {
  
  const examples = [
    {'text':'My name is Julien and I like to'},
    {'text':'My name is Thomas and my main'},
    {'text':'My name is Mariama, my favorite'}

  ]


  const [question, setQuestion] = useState(examples[0].question);
  const [text, setText] = useState(examples[0].text);
  const [jsonResponse, setJsonResponse] = useState({})
  const [responseTime, setResponseTime] = useState(0)
  const [serverTime, setServerTime] = useState(0)
  const [result, setResult] = useState(0)
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


  const exampleChange = (event) => {
    setText(examples[event.target.value].text)
    setExample(event.target.value)
  }

  const updateExample = (x) => {
    setText(examples[x].text)
  }

  const updateAnswer = (r) => {
    //setResult(answer.substring(r.results[0].start,r.results[0].end))
    setJsonResponse(r)
  }

  const query = () => {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text:text})
    };
    //console.log(requestOptions)
    var time = new Date().getTime()

    fetch("http://192.168.3.114:8100/gpt2_rabbit",requestOptions)
      .then(response => response.json())
      .then(r => {
        //setResult(r.text)
        console.log(r)
        setResult(r.results)
        setResponseTime(new Date().getTime()-time)
        setServerTime(parseInt(1000*r.server_time))
        //updateAnswer(r)
      })
    
    //.then(data => this.setState({ postId: data.id }));
  }

    return (
  <Grid container spacing={2}>
    <Grid item xs={12}>
    <h1>GPT 2 Text Generation</h1>
    </Grid>
    
    <Grid item xs={12}>
    <TextField value={text} label="text" style = {{width: 800}} multiline="True" onChange={(e) => {setText(e.target.value);}} InputLabelProps={{ shrink: true }}/>
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
      <TextField value={result} multiline="True" label="response" style = {{width: 800}}/>
    </Grid>
    <Grid item xs={12}>
      <b>API Documentation/Root : <a href="http://192.168.3.114:8100/docs" target="_blank">http://192.168.3.114:8100/docs</a></b>
    </Grid>
    <Grid item xs={12}>
      <b>Model Configuration Source <a href="https://phabricator.sourcevertex.net/diffusion/PUBLICEXAMPLES/browse/andy%252Fbert_inerference/applications/inference/project/gpt2_proto.py" target="_blank">gpt2_proto.py</a></b>
    </Grid>
    <Grid item xs={12}>
      <ReactJson collapsed="True" src={jsonResponse}/>
    </Grid>
    
  </Grid>
            
        
       

    );

}

export default Squad;
