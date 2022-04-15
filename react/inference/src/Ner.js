import TextField from '@mui/material/TextField';
import {Button, collapseClasses, Grid} from '@mui/material'
//import SquadTable from './SquadTable'
import React, { useState, useEffect } from 'react';
import SelectExample from './SelectExample';
import ReactJson from 'react-json-view'
import styles from './Ner.css';



function Ner(props) {
  

  const examples = [
    {'text':'My name is Dave and I live in Santa Clara, California'},
    {'text':'Texas, Indiana, Washington State and the District of Columbia sued Alphabet Inc’s Google on Monday over what they called deceptive location-tracking practices that invade users’ privacy.'},
    {'text':'With the ball now back in Moscow court, the West was given little immediate sign that Russian President Vladimir Putin would seek to de-escalate tensions and allay fears of a deadly new conflict.'}

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

  const spanContent = (content, index) => {
    return content + "&nbsp;" +getTag(index) + "&nbsp;"
  }

  const span = (content, index) => {
    return `<span  style="background:${getBackgroundStyle(index)};color:${getStyle(index)};border-radius: 2px">${spanContent(content,index)}</span>`
    //return `<span  style="color:${getStyle(index)};border-radius: 2px">${content}</span>`

  }

  

  const getBackgroundStyle = e => {
    //return "#ede1f0"
    switch (e) {
        case 3:
          return "#e8eddd"
        case 4:
          return "#e8eddd" 
        case 5:
          return "#ddede1"
        case 6:
          return "#ddede1" 
        case 7:
          return "#ede1f0"
        case 8:  
          return "#ede1f0"    
        default:
          return "#ede5dd" 
    }
}

  const getStyle = e => {
      switch (e) {
          case 3:
            return "#4b8536"
          case 4:
            return "#4b8536" 
          case 5:
            return "#4c4f1b"
          case 6:
            return "#4c4f1b" 
          case 7:
            return "#491b4f"
          case 8:  
            return "#491b4f"    
          default:
            return "#c1cc0c" 
      }
  }

  const getTag = e => {
    switch (e) {
        case 3:
          return perToken
        case 4:
          return perToken 
        case 5:
          return miscToken
        case 6:
          return miscToken
        case 7:
          return locToken
        case 8:  
          return locToken    
        default:
          return orgToken 
    }
    return ""
}

  const locToken = "<sup class='loc'>LOC</sup>"
  const perToken = "<sup class='per'>NAM</sup>"
  const miscToken = "<sup class='misc'>MISC</sup>"
  const orgToken = "<sup class='org'>ORG</sup>"

  const updateAnswer = (r) => {
    //array1.forEach(element => console.log(element));
    setJsonResponse(r)
    console.log(r)
    const results = r.results
    var response = ""
    if (results.length == 0) {
        response = text
        setRawMarkup(text)
    }
    else {
        var response = "<p>"
        var last = 0
        if (r.results[0].start > 0) {
            response += text.substring(0,r.results[0].start)
            last = r[0]
        }
        var index = 0
        while (index < r.results.length) {
            // code block to be executed
            var last_index = index
            if (index < r.results.length-1 && r.results[index+1].index % 2 == 0) {
              var offset = 1
              while (index + offset < r.results.length) {
                //console.log("Here", index, offset, r.results, last_index)
                if (r.results[index+offset].index % 2 == 0) {
                  last_index = last_index + 1
                  offset += 1
                }
                else {
                  //console.log("Here", index, offset, r.results, last_index)
                  break;
                }
              }
            }
            //console.log(index, last_index)

            const e = r.results[index]
            if (e.start > last) {
                response += (text.substring(last, e.start))
            }
            response += span(text.substring(e.start, r.results[last_index].end), e.index)  //"(" + e.index + ")" 
            last = r.results[last_index].end
            index =last_index + 1;
        }

        if (last < text.length) {
            response += text.substring(last,text.length)
            last = r[0]
        }
        response += "</p>"
        //setResponse( response )
        console.log("A", response)
        response=response.replaceAll("\n","<br>")
        setRawMarkup(response)
    }
    
    //console.log(r.results[0])
    //setResponse(answer.substring(r.results[0].start,r.results[0].end))
  }

  //const rawMarkup = '<b>    <mark>Hello</mark>  Ihere</b> <span style="background-color:#00FEFE">Cyan Text</span>'

  const query = () => {
    console.log(text)
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text})
    };
    console.log(requestOptions)
    var time = new Date().getTime()
    fetch("http://192.168.3.114:8100/ner_rabbit",requestOptions)
      .then(response => response.json())
      .then(r => {
          console.log(r)
          setResponseTime(new Date().getTime() - time)
          setServerTime(parseInt(1000*r.server_time))
          updateAnswer(r)
        })
    
    //.then(data => this.setState({ postId: data.id }));
  }

    return (
  <Grid container spacing={2}>
    <Grid item xs={12}><h1>BERT Large Named Entity Recognition</h1></Grid>
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
    <span dangerouslySetInnerHTML={{__html: rawMarkup}} />
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

export default Ner;
