const users = require('../models/users-model')
const express = require('express')
const router = express.Router()

router.get('/',users.getAllUsers)
router.post('/',users.createUser)

router.get('/:uid',users.getAllUserById)

router.put('/:uid',users.updateUser)
router.put('/',users.upgradeUser)
router.delete('/:uid',users.deleteUser)


module.exports = router;