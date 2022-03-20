import logo from './logo.svg';
import './App.css';
import 'react-tabs/style/react-tabs.css';
import Squad from './Squad.js'
import Ner from './Ner.js'
import GPT2 from './GPT2.js'
import Main from './Main.js'
import Bart from './Bart.js'
import Batch from './Batch.js'

import { BrowserRouter, Router, Route, Routes, Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import Button from '@mui/material/Button';
import { useNavigate } from "react-router-dom";
import Box from '@mui/material/Box';

import {
  createStyles,
  makeStyles,
  ThemeProvider,
  createTheme
} from "@mui/material/styles"
import { MenuItem } from '@mui/material';

const pages = ['Squad', 'Ner'];



function App() {


  return (
    <ThemeProvider theme={createTheme()}>
      
      <BrowserRouter>
        <AppBar position="static">
          <Container maxWidth="xl">
            <Toolbar disableGutters>
              
              <Typography>Graphcore Inference API</Typography>
              <Box sx={{ml:5}}><Typography><Link to="/">HOME</Link></Typography></Box>
              <Box sx={{ml:5}}><Typography><Link to="/squad">SQUAD</Link></Typography></Box>
              <Box sx={{ml:5}}><Typography><Link to="/ner">NER</Link></Typography></Box>
              <Box sx={{ml:5}}><Typography><Link to="/gpt2">GPT2</Link></Typography></Box>
              <Box sx={{ml:5}}><Typography><Link to="/bart">BART</Link></Typography></Box>
              <Box sx={{ml:5}}><Typography><Link to="/batch">BATCH</Link></Typography></Box>

            </Toolbar>

          </Container>
        </AppBar>
        <Box sx={{ml:5}}>
        <Routes>
          <Route path="/" element={<Main/>}/>
          <Route path="/squad" element={<Squad/>}/>
          <Route path="/ner" element={<Ner/>}/> 
          <Route path="/gpt2" element={<GPT2/>}/> 
          <Route path="/bart" element={<Bart/>}/> 
          <Route path="/batch" element={<Batch/>}/> 
        </Routes>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
    
  );
}

export default App;
