const Pool = require('pg').Pool
const pool = new Pool({
    user: "snow",
    host: "localhost",
    database: "postgres",
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

const addPredict = (req, res) => {
    const { uid, prediction_result } = req.body;
    pool.query(
        'INSERT INTO predictions (uid,prediction_result) VALUES ($1, $2) RETURNING *',
        [uid, prediction_result], (error, result) => {
            if (error) {
                throw error
            }
            res.status(201).send(`User: ${uid} with Result: ${prediction_result}`)
        }
    );
}

const getAllPredict = (req, res) => {
    pool.query('SELECT * FROM predictions', (error, result) => {
        if (error) {
            throw error
        }
        res.status(200).json(result.rows);
    })
}

const getPredictById = (req, res) => {
    const uid = parseInt(req.params.uid)
    pool.query('SELECT * FROM scores WHERE uid = $1',
        [uid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).json(result.rows);
        })
}


const deletePredict = (req, res) => {
    const uid = parseInt(req.params.uid)
    pool.query('DELETE FROM scores WHERE uid = $1',
        [uid], (error, result) => {
            if (error) {
                throw error
            }
        }
    )
    res.status(200).send(`deleted ID: ${uid}`)
}



module.exports = {
    addPredict,
    getAllPredict,
    getPredictById,
    deletePredict
}