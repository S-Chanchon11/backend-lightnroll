// const Users = require('../models/users-model');

// const getAllUsers = async (req, res) => {
//   try {
//     const user = await Users.getAllUsers();
//     res.json(user);
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// };

// const getUsersById = async (req, res) => {
//   const uid = parseInt(req.params.uid);
//   try {
//     const user = await Users.getAllUserById(uid);
//     if (!user) {
//       res.status(404).json({ error: 'user not found' });
//     } else {
//       res.json(user);
//     }
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// };

// const createUsers = async (req, res) => {
//   const { uid, dob, email, user_level, username, password  } = req.body;
//   try {
//     const user = await Users.createUser(uid, dob, email, user_level, username, password );
//     res.status(201).json(user);
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// };

// const updateUsers = async (req, res) => {
//   const id = parseInt(req.params.id);
//   const { task, completed } = req.body;
//   try {
//     const updatedTodo = await Users.update(id, task, completed);
//     if (!updatedTodo) {
//       res.status(404).json({ error: 'Todo not found' });
//     } else {
//       res.json(updatedTodo);
//     }
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// };

// const deleteUsers = async (req, res) => {
//   const id = parseInt(req.params.id);
//   try {
//     await Todo.remove(id);
//     res.sendStatus(204);
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// };

// module.exports = {
//   getAllUsers,
//   getUsersById,
//   createUsers,
//   updateUsers,
//   deleteUsers,
// };
