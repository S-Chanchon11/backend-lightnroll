const predicts = require('../models/predicts-models')
const express = require('express')
const router = express.Router()

router.get('/', predicts.getAllPredict)
router.post('/', predicts.addPredict)
router.get('/:uid', predicts.getPredictById)
router.delete('/:uid', predicts.deletePredict)

module.exports = router;