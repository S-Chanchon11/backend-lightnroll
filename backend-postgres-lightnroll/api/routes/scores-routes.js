const scores = require('../models/scores-models')
const express = require('express')
const router = express.Router()

router.get('/', scores.getAllResult)
// router.get('/rid/:rid', scores.checkResultIsExisted)
router.get('/rid/:rid', scores.getResultByRID)

router.post('/', scores.addResult)
router.put('/', scores.updateResult)
router.get('/:uid', scores.getResultById)
router.delete('/:uid', scores.deleteResult)

module.exports = router;