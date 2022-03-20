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



function Bart(props) {
  
  const examples = [
    {'text':'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.'},
    {'text':'Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital. Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well. Therefore, Peter stayed with her at the hospital for 3 days without leaving.'},
    {'text':'The East Slavs emerged as a recognisable group in Europe between the 3rd and 8th centuries AD. The medieval state of Kievan Rus arose in the 9th century. In 988, it adopted Orthodox Christianity from the Byzantine Empire. Rus ultimately disintegrated, and among its principalities, the Grand Duchy of Moscow rose. By the early 18th century, Russia had vastly expanded through conquest, annexation, and exploration to evolve into the Russian Empire, the third-largest empire in history. Following the Russian Revolution, the Russian SFSR became the largest and the principal constituent of the Soviet Union, the worlds first constitutionally socialist state. The Soviet Union played a decisive role in the Allied victory in World War II and emerged as a superpower and rival to the United States during the Cold War. The Soviet era saw some of the most significant technological achievements of the 20th century, including the worlds first human-made satellite and the launching of the first human into space.'}

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

    fetch("http://192.168.3.114:8100/bart",requestOptions)
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
    <h1>Bart Text Summarization</h1>
    <p>This model is not optimized so has poor performance. Optimization is in progess to allow similar performance to GPT2</p>
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

export default Bart;
