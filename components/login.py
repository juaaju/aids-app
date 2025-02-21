from flet import *

class LoginView(View):
    def __init__(self, page: Page, on_login_success):
        super().__init__(route="/login")
        self.page = page
        self.on_login_success = on_login_success
        
    def build(self):
        self.username = TextField(
            label='Username',
            border=InputBorder.NONE,
            filled=True,
            prefix_icon=Icons.PERSON
        )
        self.password = TextField(
            label='Password',
            password=True,
            can_reveal_password=True,
            border=InputBorder.NONE,
            filled=True,
            prefix_icon=Icons.LOCK
        )
        
        return Container(
            bgcolor=colors.WHITE,
            content=Column(
                [
                    Container(
                        alignment=alignment.center,
                        margin=margin.only(bottom=16),
                        content=Column(
                            [
                                Image(src='images/pertamina.png', width=100),
                                Text(
                                    'A I LOPE U',
                                    style=TextStyle(weight=FontWeight.W_800, color=colors.BLACK, size=50)
                                ),
                            ],
                            alignment=CrossAxisAlignment.CENTER
                        )
                    ),
                    Container(
                        width=400,
                        padding=padding.all(10),
                        content=Column(
                            [
                                Text(
                                    'Please login to start the app',
                                    style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16),
                                    text_align=TextAlign.CENTER,  # Center the text
                                ),
                                Container(  # Wrap TextField in Container for width control
                                    width=300,
                                    content=self.username,
                                ),
                                Container(  # Wrap TextField in Container for width control
                                    width=300,
                                    content=self.password,
                                ),
                                Container(  # Wrap Button in Container for width control
                                    width=300,
                                    content=FilledButton(
                                        'Login',
                                        on_click=self.login,
                                        style=ButtonStyle(
                                            color=colors.WHITE,
                                            bgcolor=colors.RED,
                                            padding=padding.symmetric(vertical=20)
                                        )
                                    )
                                )
                            ],
                            horizontal_alignment=CrossAxisAlignment.CENTER,  # Center horizontally
                            alignment=MainAxisAlignment.CENTER,  # Center vertically
                            spacing=20
                        )
                    )
                ],
                horizontal_alignment=CrossAxisAlignment.CENTER,  # Center all content horizontally
                alignment=MainAxisAlignment.CENTER,  # Center all content vertically
            ),
            expand=True,
            alignment=alignment.center  # Center the main container
        )

    def login(self, e):
        if self.username.value == 'admin' and self.password.value == 'admin':
            self.on_login_success()
        else:
            self.page.dialog = AlertDialog(title=Text("Login failed. Please try again"))
            self.page.dialog.open = True
            self.password.value = ""
            self.password.update()
            self.page.update()