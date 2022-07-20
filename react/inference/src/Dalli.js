import TextField from '@mui/material/TextField';
import {Button, collapseClasses, Grid} from '@mui/material'
//import SquadTable from './SquadTable'
import React, { useState, useEffect } from 'react';
import SelectExample from './SelectExample';
import ReactJson from 'react-json-view'
import styles from './Ner.css';

import {BASE_ADDRESS,ONLINE_ADDRESS} from './Constants'

function Dalli(props) {
  

  const examples = [
    {'text':'Avocado Chair'},
    {'text':'Artificial Intelligence'},
    {'text':'Godzilla Trial'}

  ]

  const [text, setText] = useState(examples[0].text);
  const [response, setResponse] = useState(<b>George</b>);
  const [rawMarkup, setRawMarkup] = useState();
  const [jsonResponse, setJsonResponse] = useState({})
  const [responseTime, setResponseTime] = useState(0)
  const [serverTime, setServerTime] = useState(0)

  const updateExample = (x) => {
    setText(examples[x].text)
  }

  const updateAnswer = (x) => {
    console.log("Called")
    console.log(x)
    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');
    var e = 1
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        context.fillStyle = `rgb(
          ${x[i][j][0]},
          ${x[i][j][1]},
          ${x[i][j][2]})`;
          context.fillRect(j * e, i * e, e, e);
      }
    }
    //context.rect(20, 10, 200, 100);
    //context.fillStyle = 'blue';
    //context.fill();

  }

  

  //const rawMarkup = '<b>    <mark>Hello</mark>  Ihere</b> <span style="background-color:#00FEFE">Cyan Text</span>'

  const query = () => {
    console.log(text)
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, seed:0})
    };
    console.log(requestOptions)
    var time = new Date().getTime()
    fetch(ONLINE_ADDRESS + "dalli_rabbit",requestOptions)
      .then(response => response.json())
      .then(r => {
          console.log(r)
          setResponseTime(new Date().getTime() - time)
          setServerTime(parseInt(1000*r.server_time))
          updateAnswer(r.results)
        })
    
    //.then(data => this.setState({ postId: data.id }));
  }

    return (
  <Grid container spacing={2}>
    <Grid item xs={12}><h1>Min DallE</h1></Grid>
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
    <Grid item xs={9} style={{ }}>
    <canvas id="myCanvas" width="256" height="256"></canvas>
    </Grid>
    <Grid item xs={12}>
      <b>API Documentation/Root : <a href="http://192.168.3.114:8100/docs" target="_blank">http://192.168.3.114:8100/docs</a></b>
    </Grid>
    <Grid item xs={12}>
      <b>Model Configuration Source <a href="https://phabricator.sourcevertex.net/diffusion/PUBLICEXAMPLES/browse/andy%252Fbert_inerference/applications/inference/project/ner_proto.py" target="_blank">ner_proto.py</a></b>
    </Grid>
    <Grid item xs={12}>
      <ReactJson collapsed="True" src={jsonResponse}/>
    </Grid>
    
  </Grid>
            
        
       

    );

}

export default Dalli;
