import TextField from '@mui/material/TextField';
import {Button, Grid} from '@mui/material'
//import SquadTable from './SquadTable'
import React, { useState, useEffect } from 'react';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem'
import OutlinedInput from '@mui/material/OutlinedInput';
import { useTheme } from '@mui/material/styles';




function SelectExampleBatch(props) {
  

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

      {props.exampleNames.map((name, index) => (
          <MenuItem
            key={index}
            value={index}
          >{name}
          </MenuItem>
        ))}

    </Select>

            
        
       

    );

}

export default SelectExampleBatch;
