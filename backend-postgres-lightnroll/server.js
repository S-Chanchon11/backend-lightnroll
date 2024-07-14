const express = require('express');
const app = express();
const port = 8888;

const userRoutes = require('./api/routes/users-routes')
const scoreRoutes = require('./api/routes/scores-routes')
// const predictRoutes = require('./api/routes/scores-routes')

app.use(express.json());

app.use('/users',userRoutes)
app.use('/results',scoreRoutes)
// app.use('/predicts',predictRoutes)

app.listen(port, () => {
    console.log(`Server running at http://172.20.10.3:${port}`);
});
