import TextField from '@mui/material/TextField';
import {Button, Grid} from '@mui/material'
//import SquadTable from './SquadTable'
import React, { useState, useEffect } from 'react';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem'
import OutlinedInput from '@mui/material/OutlinedInput';
import { useTheme } from '@mui/material/styles';


/*
const examples = [
    {'question':'Where do I live?','context':'My name is Wolfgang and I live in Berlin'},
    {'question':"What's my name?",'context':'My name is Clara and I live in Berkeley.'},
    {'question':'Which name is also used to describe the Amazon rainforest in English?',
    'context':'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America'}
  ]
*/

function SelectExample(props) {
  

  //const [question, setQuestion] = useState(examples[0].question);
  //const [answer, setAnswer] = useState(examples[0].context);
  const [example, setExample] = useState(0);
  const theme = useTheme();
  

  const names = [0,1,2];


  const exampleChange = (event) => {
    //setQuestion(examples[event.target.value].question)
    //setAnswer(examples[event.target.value].context)
    console.log("Example Change")
    props.updateExample(event.target.value)
    setExample(event.target.value)
    console.log(event.target.value)
  }



    return (

        <Select
        sx={{  height:36, minWidth: 80 }}
        labelId="demo-multiple-name-label"
        id="demo-multiple-name"
        
        value={example}
        onChange={exampleChange}
        input={<OutlinedInput label="Name" />}
        
    >

      {names.map((name) => (
          <MenuItem
            key={name}
            value={name}
          >Example {name+1}
          </MenuItem>
        ))}

    </Select>

            
        
       

    );

}

export default SelectExample;
