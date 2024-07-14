const Pool = require('pg').Pool
const pool = new Pool({
    user: "snow",
    host: "localhost",
    database: "postgres",
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

const createUser = (req, res) => {
    const { uid, email, user_level, username } = req.body;
    pool.query(
        'INSERT INTO users (uid,email,user_level,username) VALUES ($1, $2, $3,$4) RETURNING *',
        [uid, email, user_level, username], (error, result) => {
            if (error) {
                throw error
            }
            res.status(201).send(`User: ${uid} `)
        }
    );
}

const getAllUsers = (req, res) => {
    pool.query('SELECT * FROM users', (error, result) => {
        if (error) {
            throw error
        }
        res.status(200).json(result.rows);
    })


}

const getAllUserById = (req, res) => {
    const uid = req.params.uid
    // console.log(uid)
    pool.query('SELECT * FROM users WHERE uid = $1',
        [uid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).json(result.rows[0]);
        })
    }

const updateUser = (req, res) => {

    const _uid = parseInt(req.params.uid)
    const { username } = req.body
    pool.query('UPDATE users SET user_level = $1 WHERE uid = $2',
        [username, _uid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).send(`update user ${_uid} `)
        }
    )
}

const upgradeUser = (req, res) => {

    const { uid } = req.body
    pool.query('UPDATE users SET user_level = 2 WHERE uid = $1',
        [uid], (error, result) => {
            if (error) {
                throw error
            }
            res.status(200).send(`update user ${uid} `)
        }
    )
}

const deleteUser = (req, res) => {
    const uid = req.params.uid
    pool.query('DELETE FROM users WHERE uid = $1',
        [uid], (error, result) => {
            if (error) {
                throw error
            }
        }
    )
    res.status(200).send(`deleted ID: ${uid}`)
}



module.exports = {
    createUser,
    getAllUsers,
    getAllUserById,
    updateUser,
    deleteUser,
    upgradeUser
}