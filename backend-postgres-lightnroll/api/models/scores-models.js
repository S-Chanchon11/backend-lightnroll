
const Pool = require('pg').Pool
const pool = new Pool({
    user: "snow",
    host: "localhost",
    database: "postgres",
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

const addResult = (req, res) => {
    const { uid, rid, score,pred_result,song_name,tempo } = req.body;
    pool.query(
        'INSERT INTO results (uid,rid,score,pred_result,song_name,tempo) VALUES ($1, $2,$3,$4,$5,$6) RETURNING *',
        [uid, rid, score,pred_result,song_name,tempo], (error, result) => {
            if (error) {
                throw error
            }
            res.status(201).send(`User: ${uid} with rid: ${rid}`)
        }
    );
}
function compareLists(list1, foo) {
    // Check if lists are of the same length
    list2 = ["C","D","E","m","D","C","D","E","m","D"]
    // console.log(list1)
    // console.log(list2)
    if (list1.length !== list2.length) {
        throw new Error("Lists are not of the same length and cannot be compared");
    }

    // Initialize a counter for differences
    let differencesCount = 0;

    // Iterate over the lists and compare elements at corresponding positions
    for (let i = 0; i < list1.length; i++) {
        if (list1[i] !== list2[i]) {
            differencesCount++;
        }
    }

    // Calculate the percentage of differences
    let differencePercentage = (differencesCount / list1.length) * 100;
    let diff = 100-differencePercentage
    let finalResult = parseFloat(diff.toFixed(2))

    // Return the percentage of differences
    return finalResult
}

const getChordList = (req, res) => {
    pool.query(
        'SELECT chord_list FROM songs WHERE song_name = $1',
        [ rid, pred_result,accPercentage], (error, result) => {
            if (error) {
                throw error
            }
            
            res.status(201).send(`rid: ${rid} with result of: ${pred_result}`)
        }
    );
}

const updateResult = async (req, res) => {
    const { rid, song_name,pred_result,detected_tempo } = req.body;
    console.log(req.body)
    await pool.query(
        'SELECT chord_list FROM songs WHERE song_name = $1',
        [song_name]).then( data => {
            newData = data.rows[0]
            accPercentage = compareLists(pred_result,newData)
            pool.query(
                'UPDATE results SET pred_result = $2, score = $3,tempo = $4 WHERE rid = $1 RETURNING rid',
                [ rid, pred_result,accPercentage,detected_tempo], (error, result) => {
                    if (error) {
                        throw error
                    }
                    
                    res.status(201).send(result.rows[0])
                }
            );
        }

        )
    
            
     
    
}

const getAllResult = (req, res) => {

    pool.query('SELECT * FROM results', (error, result) => {
        if (error) {
            throw error
        }
        res.status(200).json(result.rows);
    })
}

const getResultById = (req, res) => {
    const uid = req.params.uid
    pool.query('SELECT * FROM results WHERE uid = $1',
        [uid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).json(result.rows);
        })
}
const getResultByRID = (req, res) => {
    const rid = req.params.rid
    pool.query('SELECT * FROM results WHERE rid = $1',
        [rid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).json(result.rows[0]);
        })
}


const deleteResult = (req, res) => {
    const rid = req.params.rid
    pool.query('DELETE FROM results WHERE rid = $1',
        [rid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).send(`deleted ID: ${rid}`)
        }
        
    )
    
}

const checkResultIsExisted = (req, res) => {
    const rid = req.params.rid
    pool.query('SELECT EXISTS(SELECT 1 FROM results WHERE rid = $1)',
        [rid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).json(result.rows[0]);
        }
    )
    
}


module.exports = {
    addResult,
    getAllResult,
    updateResult,
    getResultById,
    deleteResult,
    checkResultIsExisted,
    getResultByRID
}