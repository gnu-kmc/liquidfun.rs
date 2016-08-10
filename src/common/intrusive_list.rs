extern crate std;

/// `IntrusiveListNode` is used to implement of intrusive doubly-linked
/// list.
///
/// For example:
///
/// ```
/// struct MyStruct {
///     node: IntrusiveListNode,
///     msg: &str,
/// }
///
/// impl MyStruct {
///     fn new(m: &str) -> MyStruct {
///         MyStruct{msg: m}
///     }
///
///     fn<'a> get_message(&self) -> &'a str {
///         self.msg
///     }
/// }
///
/// fn main() {
///     let mut list = IntrusiveListNode::new();
///     let a = MyStruct
/// }
/// ```
struct IntrusiveListNode {
    /// The next node in the list.
    prev: *mut IntrusiveListNode,
    /// The previous node in the list.
    next: *mut IntrusiveListNode,
}

impl IntrusiveListNode {
    /// Initialize the node.
    pub fn new() -> IntrusiveListNode {
        let mut node = IntrusiveListNode{
            prev: std::ptr::null_mut(),
            next: std::ptr::null_mut(),
        };
        node.initialize();
        node
    }

    /// Insert this node after the specified node.
    pub fn insert_after(&mut self, node: *mut IntrusiveListNode) {
        unsafe{
            (*node).next = self.next;
            (*node).prev = self as *mut IntrusiveListNode;
            (*self.next).prev = node;
            self.next = node;
        }
    }

    pub fn insert_before(&mut self, node: *mut IntrusiveListNode) {
        unsafe{
            (*node).next = self as *mut IntrusiveListNode;
            (*node).prev = self.prev;
            (*self.prev).next = node;
            self.prev = node;
        }
    }

    pub fn get_terminator(&mut self) -> *mut IntrusiveListNode {
        self
    }

    pub fn remove(&mut self) -> *mut IntrusiveListNode {
        unsafe{
            (*self.prev).next = self.next;
            (*self.next).prev = self.prev;
        }
        self.initialize();
        self
    }

    /// Determine whether this list is empty or the node isn't in a list.
    pub fn is_empty(&mut self) -> bool {
        self.get_next() == self as *mut IntrusiveListNode
    }

    /// Determine whether this node is in a list or the list contains nodes.
    pub fn in_list(&mut self) -> bool {
        !self.is_empty()
    }

    /// Calculate the length of the list.
    pub fn get_length(&mut self) -> u32 {
        let mut length: u32 = 0;
        let terminator = self.get_terminator();
        let mut node = self.get_next();
        while node != terminator {
            length += 1;
            node = unsafe{(*node).get_next()}
        }
        length
    }

    /// Get the next node in the list.
    pub fn get_next(&self) -> *mut IntrusiveListNode {
        self.next
    }

    /// Get the previous node in the list.
    pub fn get_previous(&self) -> *mut IntrusiveListNode {
        self.prev
    }

    /// If INTRUSIVE_LIST_VALIDATE is 1 perform a very rough validation
    /// of all nodes in the list.
    pub fn validate_list(&self) -> bool {
        true
    }

    /// Determine whether the specified node is present in this list.
    pub fn find_node_in_list(&mut self, node_to_find: *const IntrusiveListNode) -> bool {
        let terminator = self.get_terminator();
        let mut node = self.get_next();
        while node != terminator {
            if node_to_find == node {
                return true;
            }
            node = unsafe{(*node).get_next()};
        }
        return false;
    }

    /// Initialize th list node.
    fn initialize(&mut self) {
        self.next = self as *mut IntrusiveListNode;
        self.prev = self as *mut IntrusiveListNode;
    }
}

/// Declares the member function get_lis_node() of Class to retrieve a pointer
/// to node_member_name.
/// See #intrusive_list_node_get_class_accessor()
macro_rules! intrusive_list_get_node {
    ($node_member_name:ident) => {
        fn get_list_node(&mut self) -> *mut IntrusiveListNode {
            &mut self.$node_member_name as *mut IntrusiveListNode
        }
        fn node(&self) -> &IntrusiveListNode {
            &self.$node_member_name
        }
    }
}

trait GetListNode {
    fn get_list_node(&mut self) -> *mut IntrusiveListNode;
    fn node(&self) -> &IntrusiveListNode;
}

/// Declares the member function function_name of class to retrieve a pointer
/// to a Class instance from a list node pointer. node_member_name references
/// the name of the Intrusive member of class.
macro_rules! intrusive_list_node_get_class_accessor {
    ($class:ident, $node_member_name:ident, $function_name:ident) => {
        fn $function_name(&mut self, node: *mut IntrusiveListNode) -> $class {
            let mut cls: *mut $class = std::mem::null_mut();
            /* This effectively performs offsetof($class, $node_member_name) */
            /* which ends up in the undefined behavior realm of C++ but in */
            /* practice this works with most compilers. */
            unsafe{std::mem::transmute_copy::<*const u8, $class>(std::mem::transmute_copy::<*const IntrusiveListNode, *const u8>(&node) - std::mem::transmute_copy::<*const IntrusiveListNode, *const u8>(&cls.$node_member_name))}
        }
    }
}

/// Declares the member function get_instance_from_list_node() of $class to retrieve
/// a pointer to a $class instance from a list node pointer. $node_member_name
/// reference the name of the IntrusiveListNode member of $class.
macro_rules! intrusive_list_node_get_class {
    ($class:ident, $node_member_name:ident) => {
        intrusive_list_node_get_class_accessor!($class, $node_member_name, get_instance_from_list_node)
    }
}

/// TypedIntrusiveListNode which supports inserting an object into a single
/// doubly linked list. For objects that need to be inserted in multiple
/// doubly linked lists, use IntrusiveListNode.
///
/// For example:
///
struct TypedIntrusiveListNode<T> where T: GetListNode {
    node: IntrusiveListNode,
    phantom: std::marker::PhantomData<T>,
}

impl<T> TypedIntrusiveListNode<T> where T: GetListNode {
    pub fn new() -> TypedIntrusiveListNode<T> {
        TypedIntrusiveListNode::<T>{
            node: IntrusiveListNode::new(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Insert this object after the specified object.
    pub fn insert_after(&mut self, obj: *mut T) {
        assert!(obj != std::ptr::null_mut::<T>());
        unsafe{(*self.get_list_node()).insert_after((*obj).get_list_node())};
    }

    /// Inseert this object before the specified object.
    pub fn insert_before(&mut self, obj: *mut T) {
        assert!(obj != std::ptr::null_mut::<T>());
        unsafe{(*self.get_list_node()).insert_before((*obj).get_list_node())};
    }

    /// Get the next object in the list.
    /// Check against get_terminator() before deferencing the object.
    pub fn get_next(&mut self) -> *mut T {
        unsafe{
            let ref mut a = *self.get_list_node();
            self.get_instance_from_list_node(a.get_next())
        }
    }

    /// Get the previous object in the list.
    /// Check against get_terminator() before deferencing th object.
    pub fn get_previous(&mut self) -> *mut T {
        unsafe{
            let ref mut a = *self.get_list_node();
            self.get_instance_from_list_node(a.get_previous())
        }
    }

    /// Get the terminator of the list.
    /// This sould not be dereferenced as it is a pointer to
    /// TypedIntrusiveListNode<T> *not* T.
    pub fn get_terminator(&mut self) -> *mut T {
        self.get_list_node() as *mut T
    }

    /// Remove this object from the list it's currently in.
    pub fn remove(&mut self) -> *mut T {
        let a = self.get_list_node();
        self.get_instance_from_list_node(a)
    }

    /// Determine whether this object is in a list.
    pub fn in_list(&mut self) -> bool {
        unsafe{(*self.get_list_node()).in_list()}
    }

    /// Determine whether this list is empty.
    pub fn is_empty(&mut self) -> bool {
        unsafe{(*self.get_list_node()).is_empty()}
    }

    /// Calculate the length of the list.
    pub fn get_length(&mut self) -> u32 {
        unsafe{(*self.get_list_node()).get_length()}
    }

    /// Get a pointer to the instnce of T that contains "node".
    pub fn get_instance_from_list_node(&self, node: *mut IntrusiveListNode) -> *mut T {
        assert!(node != std::ptr::null_mut::<IntrusiveListNode>());
        // Calculate the pointer to T from the offset.
        unsafe{std::mem::transmute_copy::<i32, *mut T>(&(std::mem::transmute_copy::<*mut IntrusiveListNode, i32>(&node) - self.get_node_offset(node)))}
    }

    // Get the offset of node within this class.
    fn get_node_offset(&self, node: *mut IntrusiveListNode) -> i32 {
        assert!(node != std::ptr::null_mut::<IntrusiveListNode>());
        // Perform some type punning to calculate the offset of node in T.
        // WARNING: This could result in undefined behavior with some C++
        // compilers.
        let obj: *mut T = node as *mut T;
        unsafe{std::mem::transmute_copy::<*const IntrusiveListNode, i32>(&((*obj).node() as *const IntrusiveListNode)) - std::mem::transmute_copy::<*const T, i32>(&(obj as *const T))}
    }
}

impl<T: GetListNode> GetListNode for TypedIntrusiveListNode<T> {
    intrusive_list_get_node!(node);
}
